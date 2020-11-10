//! How to use this crate
//! # Adding this as a dependency
//! ```rust, ignore
//! [dependencies]
//! ssvm_tensorflow_interface = "^0.0.1"
//! ```
//!
//! # Bringing this into scope
//! ```rust, ignore
//! use ssvm_tensorflow_interface;
//! ```

use std::ffi::CString;
use std::mem;
use std::os::raw::c_char;
use std::str;
#[link(wasm_import_module = "ssvm_tensorflow")]
extern "C" {
    pub fn ssvm_tensorflow_load_jpg_to_rgb8(
        img_buf: *const c_char,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_jpg_to_bgr8(
        img_buf: *const c_char,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_jpg_to_rgb32f(
        img_buf: *const c_char,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_jpg_to_bgr32f(
        img_buf: *const c_char,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_rgb8(
        img_buf: *const c_char,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_bgr8(
        img_buf: *const c_char,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_rgb32f(
        img_buf: *const c_char,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_bgr32f(
        img_buf: *const c_char,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_run_vision(
        model_buf: *const c_char,
        model_buf_len: u32,
        tensor_buf: *const c_char,
        tensor_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_append_input(input_name: *const c_char, input_name_len: u32);
    pub fn ssvm_tensorflow_append_output(output_name: *const c_char, output_name_len: u32);
    pub fn ssvm_tensorflow_get_result_len() -> u32;
    pub fn ssvm_tensorflow_get_result(buf: *mut u8);
}

/// Convert JPEG image in memory into rgb u8 vector.
pub fn load_jpg_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_rgb8(img_buf.as_ptr() as *const i8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr);
        result_vec
    }
}

/// Convert JPEG image in memory into bgr u8 vector.
pub fn load_jpg_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_bgr8(img_buf.as_ptr() as *const i8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr);
        result_vec
    }
}

/// Convert JPEG image in memory into rgb f32 vector.
pub fn load_jpg_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_rgb32f(
            img_buf.as_ptr() as *const i8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr as *mut u8);
        result_vec
    }
}

/// Convert JPEG image in memory into bgr f32 vector.
pub fn load_jpg_image_to_bgr32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_bgr32f(
            img_buf.as_ptr() as *const i8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr as *mut u8);
        result_vec
    }
}

/// Convert PNG image in memory into rgb u8 vector.
pub fn load_png_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_png_to_rgb8(img_buf.as_ptr() as *const i8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr);
        result_vec
    }
}

/// Convert PNG image in memory into bgr u8 vector.
pub fn load_png_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_png_to_bgr8(img_buf.as_ptr() as *const i8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr);
        result_vec
    }
}

/// Convert PNG image in memory into rgb f32 vector.
pub fn load_png_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_png_to_rgb32f(
            img_buf.as_ptr() as *const i8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr as *mut u8);
        result_vec
    }
}

/// Convert PNG image in memory into bgr f32 vector.
pub fn load_png_image_to_bgr32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_png_to_bgr32f(
            img_buf.as_ptr() as *const i8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr as *mut u8);
        result_vec
    }
}

/// Run tensorflow model with image input tensor.
pub fn run_tensorflow_vision(
    model_buf: &[u8],
    tensor_buf: &[f32],
    image_width: u32,
    image_height: u32,
    input_name: &str,
    output_names: &[&str],
) -> String {
    unsafe {
        let input_name_cstr = CString::new(input_name).expect("");
        ssvm_tensorflow_append_input(
            input_name_cstr.as_ptr(),
            input_name_cstr.as_bytes().len() as u32,
        );

        for s in output_names.iter() {
            let output_name_cstr = CString::new(*s).expect("");
            ssvm_tensorflow_append_output(
                output_name_cstr.as_ptr(),
                output_name_cstr.as_bytes().len() as u32,
            );
        }
        ssvm_tensorflow_run_vision(
            model_buf.as_ptr() as *const i8,
            model_buf.len() as u32,
            tensor_buf.as_ptr() as *const i8,
            (tensor_buf.len() * mem::size_of::<f32>()) as u32,
            image_width,
            image_height,
        );

        let result_len = ssvm_tensorflow_get_result_len();
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(result_ptr);
        String::from_utf8(result_vec).unwrap()
    }
}
