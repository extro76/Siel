use siel_core::QueryResponse;

/// Minimal FFI-safe smoke function for the Flutter/Rust bridge spike.
///
/// The full app should prefer typed `flutter_rust_bridge` generated bindings.
/// This function exists to validate mobile linker setup before the UI expands.
#[no_mangle]
pub extern "C" fn siel_ffi_version() -> u32 {
    1
}

pub fn offline_unknown_json(reason: &str) -> String {
    serde_json::to_string(&QueryResponse::unknown(reason))
        .unwrap_or_else(|_| "{\"status\":\"unknown\"}".to_string())
}

