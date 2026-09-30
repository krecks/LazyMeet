# Forces the static MSVC runtime (/MT) for every cmake-built native dep on
# Windows. sherpa-onnx's prebuilt static libs (onnxruntime included) use /MT;
# llama.cpp silently ignores the cmake crate's static_crt knob because its
# cmake_minimum_required predates policy CMP0091, leaving it on /MD — and the
# two runtimes cannot link into one binary (LNK2038).
#
# Injected via the CMAKE_TOOLCHAIN_FILE environment variable in CI (honored
# by CMake >= 3.21); toolchain files run before project(), early enough to
# set the policy default.
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
