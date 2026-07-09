// Prevents an extra console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{Emitter, Manager, RunEvent, State};

/// Holds the spawned Python backend's port (once it reports READY) and the
/// child handle so we can terminate it cleanly on exit.
#[derive(Default)]
struct Backend {
    port: Mutex<Option<u16>>,
    child: Mutex<Option<Child>>,
}

/// The frontend calls this to discover the backend port, waiting until the
/// backend has started (it prints `READY <port>` on stdout).
#[tauri::command]
async fn get_backend_port(state: State<'_, Backend>) -> Result<u16, String> {
    for _ in 0..240 {
        if let Some(port) = *state.port.lock().unwrap() {
            return Ok(port);
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    Err("backend did not start in time".into())
}

/// The backend ships as a PyInstaller onedir tree in the app's resources (a
/// onefile binary would re-extract ~70 MB on every launch, delaying startup by
/// many seconds).
fn backend_exe(app: &tauri::App) -> std::path::PathBuf {
    let name = format!("lazymeet-backend{}", std::env::consts::EXE_SUFFIX);
    if let Ok(dir) = app.path().resource_dir() {
        let bundled = dir.join("binaries").join("backend").join(&name);
        if bundled.exists() {
            return bundled;
        }
    }
    // Dev run: use the tree built by scripts/build_sidecar.py directly.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("binaries")
        .join("backend")
        .join(name)
}

fn main() {
    let app = tauri::Builder::default()
        .manage(Backend::default())
        .invoke_handler(tauri::generate_handler![get_backend_port])
        .setup(|app| {
            let handle = app.handle().clone();

            let mut child = Command::new(backend_exe(app))
                // The backend watches this PID and tears its own process tree
                // (server + worker subprocesses) down when we exit.
                .env("LAZYMEET_PARENT_PID", std::process::id().to_string())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .expect("failed to spawn backend");

            let stdout = child.stdout.take().expect("backend stdout not captured");
            *app.state::<Backend>().child.lock().unwrap() = Some(child);

            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    if let Some(rest) = line.trim().strip_prefix("READY ") {
                        if let Ok(port) = rest.trim().parse::<u16>() {
                            let state = handle.state::<Backend>();
                            *state.port.lock().unwrap() = Some(port);
                            let _ = handle.emit("backend-ready", port);
                        }
                    }
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building LazyMeet");

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { .. } | RunEvent::Exit = event {
            // Terminate the backend when the app closes.
            if let Some(mut child) = app_handle.state::<Backend>().child.lock().unwrap().take() {
                let _ = child.kill();
            }
        }
    });
}
