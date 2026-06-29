use std::io::{BufWriter, Write};
use std::path::Path;

use sim::TubeMesh;

/// Writes the Solid-mode tube mesh as a Wavefront OBJ file: positions, one
/// vertex normal per position (no texture coordinates -- Solid mode has no
/// UV mapping), and triangle faces. Streamed through a `BufWriter` rather
/// than built up in memory first, since meshes here can run into the tens
/// of millions of vertices.
pub fn export_obj(mesh: &TubeMesh, path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# Exported from Strange Attractor")?;
    writeln!(w, "# {} vertices, {} triangles", mesh.vertices.len(), mesh.indices.len() / 3)?;

    for v in &mesh.vertices {
        writeln!(w, "v {} {} {}", v.position[0], v.position[1], v.position[2])?;
    }
    for v in &mesh.vertices {
        writeln!(w, "vn {} {} {}", v.normal[0], v.normal[1], v.normal[2])?;
    }
    // OBJ indices are 1-based. Position and normal indices coincide since
    // each TubeVertex already carries its own normal.
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] + 1, tri[1] + 1, tri[2] + 1);
        writeln!(w, "f {a}//{a} {b}//{b} {c}//{c}")?;
    }

    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn writes_expected_line_counts() {
        let centerline = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 1.0, 1.0),
        ];
        let mesh = sim::build_tube_mesh(&centerline, 0.1, 6);
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());

        let path = std::env::temp_dir().join("mesh_export_test.obj");
        export_obj(&mesh, &path).expect("export should succeed");

        let contents = std::fs::read_to_string(&path).expect("file should be readable");
        let v_count = contents.lines().filter(|l| l.starts_with("v ")).count();
        let vn_count = contents.lines().filter(|l| l.starts_with("vn ")).count();
        let f_count = contents.lines().filter(|l| l.starts_with("f ")).count();

        assert_eq!(v_count, mesh.vertices.len());
        assert_eq!(vn_count, mesh.vertices.len());
        assert_eq!(f_count, mesh.indices.len() / 3);

        // Spot-check a face line references valid (1-based) vertex indices.
        let first_face = contents.lines().find(|l| l.starts_with("f ")).unwrap();
        for token in first_face.trim_start_matches("f ").split(' ') {
            let idx: usize = token.split("//").next().unwrap().parse().unwrap();
            assert!(idx >= 1 && idx <= mesh.vertices.len());
        }

        std::fs::remove_file(&path).ok();
    }
}
