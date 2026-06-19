use std::path::{Path, PathBuf};
use std::process::Command;

use crate::state::{self, SceneState};

const FILENAME_PREFIX: &str = "frame";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputKind {
    PngSequence,
    Gif,
    Mp4,
}

#[derive(Clone)]
pub struct MovieSettings {
    pub frames_per_step: u32,
    pub iters_per_frame: u64,
    pub loop_back: bool,
    pub output_kind: OutputKind,
    pub fps: u32,
}

#[derive(Clone)]
pub enum MovieStatus {
    Rendering { frame_index: u32, total_frames: u32 },
    Encoding,
    Done(PathBuf),
    Error(String),
    Cancelled,
}

/// Drives a sequence of independently-rendered, interpolated frames from a
/// clean histogram each time. Owns the loaded keyframes so the source files
/// can be deleted/edited mid-render without affecting the job.
pub struct MovieJob {
    keyframes: Vec<SceneState>,
    settings:  MovieSettings,

    work_dir:         PathBuf,
    is_temp_work_dir: bool,
    output_path:      PathBuf,

    total_frames: u32,
    frame_index:  u32,
    pair_index:   usize,
    step_in_pair: u32,
    frame_started: bool,

    status: MovieStatus,
}

impl MovieJob {
    pub fn new(keyframes: Vec<SceneState>, settings: MovieSettings, output_path: PathBuf) -> Result<Self, String> {
        if keyframes.len() < 2 {
            return Err("Select at least 2 keyframe state files.".to_string());
        }
        if settings.frames_per_step == 0 {
            return Err("Frames per step must be at least 1.".to_string());
        }
        state::validate_compatible(&keyframes)?;

        // Guard against a stale or manually mistyped extension producing a
        // file whose contents don't match its name (e.g. GIF data saved as
        // "movie.mpeg") — force the extension to match the chosen output kind.
        let output_path = match settings.output_kind {
            OutputKind::PngSequence => output_path,
            OutputKind::Gif => with_extension(output_path, "gif"),
            OutputKind::Mp4 => with_extension(output_path, "mp4"),
        };

        if settings.output_kind == OutputKind::Mp4 && !ffmpeg_available() {
            return Err("ffmpeg was not found on PATH — install ffmpeg or choose a different output type.".to_string());
        }

        let segments = keyframes.len() - 1 + if settings.loop_back { 1 } else { 0 };
        let total_frames = settings.frames_per_step * segments as u32;

        let (work_dir, is_temp_work_dir) = match settings.output_kind {
            OutputKind::PngSequence => (output_path.clone(), false),
            OutputKind::Gif | OutputKind::Mp4 => {
                let millis = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                (std::env::temp_dir().join(format!("attractor_movie_{millis}")), true)
            }
        };
        std::fs::create_dir_all(&work_dir).map_err(|e| format!("Failed to create working directory: {e}"))?;

        Ok(Self {
            keyframes,
            settings,
            work_dir,
            is_temp_work_dir,
            output_path,
            total_frames,
            frame_index: 0,
            pair_index: 0,
            step_in_pair: 0,
            frame_started: false,
            status: MovieStatus::Rendering { frame_index: 0, total_frames },
        })
    }

    pub fn status(&self) -> &MovieStatus { &self.status }
    pub fn frame_started(&self) -> bool { self.frame_started }
    pub fn mark_frame_started(&mut self) { self.frame_started = true; }

    /// Interpolated scene for the frame currently being accumulated.
    pub fn current_interpolated_state(&self) -> SceneState {
        let last = self.keyframes.len() - 1;
        let (a, b) = if self.settings.loop_back && self.pair_index == last {
            (&self.keyframes[last], &self.keyframes[0])
        } else {
            (&self.keyframes[self.pair_index], &self.keyframes[self.pair_index + 1])
        };
        let t = self.step_in_pair as f32 / self.settings.frames_per_step as f32;
        state::interpolate(a, b, t)
    }

    pub fn frame_filename(&self) -> PathBuf {
        self.work_dir.join(format!("{FILENAME_PREFIX}_{:05}.png", self.frame_index))
    }

    pub fn iter_target_reached(&self, iter_count: u64) -> bool {
        iter_count >= self.settings.iters_per_frame
    }

    /// Moves to the next frame. Returns `true` once every frame has been
    /// rendered, at which point the caller should call [`run_encode`].
    pub fn advance_frame(&mut self) -> bool {
        self.frame_index += 1;
        self.frame_started = false;
        self.step_in_pair += 1;
        if self.step_in_pair >= self.settings.frames_per_step {
            self.step_in_pair = 0;
            self.pair_index += 1;
        }
        let done = self.frame_index >= self.total_frames;
        self.status = if done {
            MovieStatus::Encoding
        } else {
            MovieStatus::Rendering { frame_index: self.frame_index, total_frames: self.total_frames }
        };
        done
    }

    pub fn fail(&mut self, message: String) {
        self.cleanup_temp_dir();
        self.status = MovieStatus::Error(message);
    }

    pub fn cancel(&mut self) {
        if !matches!(self.status, MovieStatus::Cancelled) {
            self.cleanup_temp_dir();
            self.status = MovieStatus::Cancelled;
        }
    }

    fn cleanup_temp_dir(&self) {
        if self.is_temp_work_dir {
            let _ = std::fs::remove_dir_all(&self.work_dir);
        }
    }

    /// All frames are on disk (status == Encoding) — run the PNG sequence's
    /// optional GIF/MP4 post-process and move to a terminal status.
    pub fn run_encode(&mut self) {
        let result = match self.settings.output_kind {
            OutputKind::PngSequence => Ok(()),
            OutputKind::Gif => encode_gif(&self.work_dir, self.total_frames, self.settings.fps, &self.output_path),
            OutputKind::Mp4 => encode_mp4(&self.work_dir, self.settings.fps, &self.output_path),
        };
        match result {
            Ok(()) => {
                self.cleanup_temp_dir();
                self.status = MovieStatus::Done(self.output_path.clone());
            }
            Err(e) => self.fail(e),
        }
    }
}

fn with_extension(path: PathBuf, ext: &str) -> PathBuf {
    let matches = path.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext));
    if matches { path } else { path.with_extension(ext) }
}

pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn encode_gif(work_dir: &Path, frame_count: u32, fps: u32, output: &Path) -> Result<(), String> {
    let file = std::fs::File::create(output).map_err(|e| e.to_string())?;
    let mut encoder = image::codecs::gif::GifEncoder::new_with_speed(file, 10);
    encoder.set_repeat(image::codecs::gif::Repeat::Infinite).map_err(|e| e.to_string())?;
    let delay = image::Delay::from_numer_denom_ms(1000, fps.max(1));
    for i in 0..frame_count {
        let path = work_dir.join(format!("{FILENAME_PREFIX}_{i:05}.png"));
        let img = image::open(&path).map_err(|e| format!("reading {}: {e}", path.display()))?.into_rgba8();
        let frame = image::Frame::from_parts(img, 0, 0, delay);
        encoder.encode_frame(frame).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn encode_mp4(work_dir: &Path, fps: u32, output: &Path) -> Result<(), String> {
    let pattern = work_dir.join(format!("{FILENAME_PREFIX}_%05d.png"));
    let out = Command::new("ffmpeg")
        .args(["-y", "-framerate", &fps.to_string(), "-i"])
        .arg(&pattern)
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(output)
        .output()
        .map_err(|e| format!("failed to launch ffmpeg: {e}"))?;
    if !out.status.success() {
        return Err(format!("ffmpeg exited with {}: {}", out.status, String::from_utf8_lossy(&out.stderr)));
    }
    Ok(())
}
