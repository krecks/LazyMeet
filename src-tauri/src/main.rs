// Prevents an extra console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use tauri::{Emitter, Manager, RunEvent, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Holds the spawned Python sidecar's port (once it reports READY) and the child
/// handle so we can terminate it cleanly on exit.
#[derive(Default)]
struct Backend {
    port: Mutex<Option<u16>>,
    child: Mutex<Option<CommandChild>>,
}

/// The frontend calls this to discover the backend port, waiting until the
/// sidecar has started (it prints `READY <port>` on stdout).
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

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(Backend::default())
        .invoke_handler(tauri::generate_handler![get_backend_port])
        .setup(|app| {
            let handle = app.handle().clone();

            let sidecar = app
                .shell()
                .sidecar("lazymeet-backend")
                .expect("failed to create sidecar command");
            let (mut rx, child) = sidecar.spawn().expect("failed to spawn backend sidecar");

            // Keep the child handle so it isn't dropped (and so we can kill it).
            *app.state::<Backend>().child.lock().unwrap() = Some(child);

            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            let text = String::from_utf8_lossy(&line);
                            for l in text.lines() {
                                if let Some(rest) = l.trim().strip_prefix("READY ") {
                                    if let Ok(port) = rest.trim().parse::<u16>() {
                                        let state = handle.state::<Backend>();
                                        *state.port.lock().unwrap() = Some(port);
                                        let _ = handle.emit("backend-ready", port);
                                    }
                                }
                            }
                        }
                        CommandEvent::Stderr(line) => {
                            eprintln!("[backend] {}", String::from_utf8_lossy(&line));
                        }
                        _ => {}
                    }
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building LazyMeet");

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { .. } | RunEvent::Exit = event {
            // Terminate the backend sidecar when the app closes.
            if let Some(child) = app_handle.state::<Backend>().child.lock().unwrap().take() {
                let _ = child.kill();
            }
        }
    });
}
