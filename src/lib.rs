//! How to use this crate
//! # Adding this as a dependency
//! ```rust, ignore
//! [dependencies]
//! ssvm_tensorflow_interface = "^0.1.1"
//! ```
//!
//! # Bringing this into scope
//! ```rust, ignore
//! use ssvm_tensorflow_interface;
//! ```

use std::ffi::CString;
use std::mem;
use std::str;
#[link(wasm_import_module = "ssvm_tensorflow")]
extern "C" {
    pub fn ssvm_tensorflow_load_jpg_to_rgb8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_jpg_to_bgr8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_jpg_to_rgb32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_jpg_to_bgr32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_rgb8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_bgr8(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_rgb32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_load_png_to_bgr32f(
        img_buf: *const u8,
        img_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_run_vision(
        model_buf: *const u8,
        model_buf_len: u32,
        tensor_buf: *const u8,
        tensor_buf_len: u32,
        img_width: u32,
        img_height: u32,
    ) -> u32;
    pub fn ssvm_tensorflow_append_input(
        input_name: *const u8,
        input_name_len: u32,
        dim_vec: *const u8,
        dim_cnt: u32,
        index: u32,
    );
    pub fn ssvm_tensorflow_append_output(output_name: *const u8, output_name_len: u32, index: u32);
    pub fn ssvm_tensorflow_clear_input();
    pub fn ssvm_tensorflow_clear_output();
    pub fn ssvm_tensorflow_get_result_len(idx: u32) -> u32;
    pub fn ssvm_tensorflow_get_result(idx: u32, buf: *mut u8);
}

/// The output tensor structure.
pub struct Tensors {
    pub data: Vec<Vec<u8>>,
}

impl Tensors {
    pub fn convert_to_vec<T>(&self, idx: usize) -> Vec<T> {
        if idx > self.data.len() {
            return Vec::new();
        }
        let v = mem::ManuallyDrop::new(&self.data[idx]);
        let p = v.as_ptr();
        unsafe {
            Vec::from_raw_parts(
                p as *mut T,
                v.len() / mem::size_of::<T>(),
                v.len() / mem::size_of::<f32>(),
            )
        }
    }
}

/// Convert JPEG image in memory into rgb u8 vector.
pub fn load_jpg_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_rgb8(img_buf.as_ptr() as *const u8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len(0);
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(0, result_ptr);
        result_vec
    }
}

/// Convert JPEG image in memory into bgr u8 vector.
pub fn load_jpg_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_bgr8(img_buf.as_ptr() as *const u8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len(0);
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(0, result_ptr);
        result_vec
    }
}

/// Convert JPEG image in memory into rgb f32 vector.
pub fn load_jpg_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_rgb32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len(0);
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(0, result_ptr as *mut u8);
        result_vec
    }
}

/// Convert JPEG image in memory into bgr f32 vector.
pub fn load_jpg_image_to_bgr32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_jpg_to_bgr32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len(0);
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(0, result_ptr as *mut u8);
        result_vec
    }
}

/// Convert PNG image in memory into rgb u8 vector.
pub fn load_png_image_to_rgb8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_png_to_rgb8(img_buf.as_ptr() as *const u8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len(0);
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(0, result_ptr);
        result_vec
    }
}

/// Convert PNG image in memory into bgr u8 vector.
pub fn load_png_image_to_bgr8(img_buf: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        ssvm_tensorflow_load_png_to_bgr8(img_buf.as_ptr() as *const u8, img_buf.len() as u32, w, h);
        let result_len = ssvm_tensorflow_get_result_len(0);
        let mut result_vec: Vec<u8> = vec![0; result_len as usize];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(0, result_ptr);
        result_vec
    }
}

/// Convert PNG image in memory into rgb f32 vector.
pub fn load_png_image_to_rgb32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_png_to_rgb32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len(0);
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(0, result_ptr as *mut u8);
        result_vec
    }
}

/// Convert PNG image in memory into bgr f32 vector.
pub fn load_png_image_to_bgr32f(img_buf: &[u8], w: u32, h: u32) -> Vec<f32> {
    unsafe {
        ssvm_tensorflow_load_png_to_bgr32f(
            img_buf.as_ptr() as *const u8,
            img_buf.len() as u32,
            w,
            h,
        );
        let result_len = ssvm_tensorflow_get_result_len(0);
        let mut result_vec: Vec<f32> = vec![0.0; result_len as usize / mem::size_of::<f32>()];
        let result_ptr = result_vec.as_mut_ptr();
        ssvm_tensorflow_get_result(0, result_ptr as *mut u8);
        result_vec
    }
}

/// Run tensorflow model with image input tensor.
pub fn run_tensorflow_vision(
    model_buf: &[u8],
    tensor_buf: &[f32],
    tensor_dim: &[i64],
    image_width: u32,
    image_height: u32,
    input_name: &str,
    output_names: &[&str],
) -> Tensors {
    unsafe {
        let input_name_pair: Vec<&str> = input_name.split(":").collect();
        let input_name_cstr = CString::new(input_name_pair[0]).expect("");
        let input_name_idx = if input_name_pair.len() > 1 {
            input_name_pair[1].parse().unwrap()
        } else {
            0
        };

        ssvm_tensorflow_append_input(
            input_name_cstr.as_ptr() as *const u8,
            input_name_cstr.as_bytes().len() as u32,
            tensor_dim.as_ptr() as *const u8,
            tensor_dim.len() as u32,
            input_name_idx,
        );

        for s in output_names.iter() {
            let output_name_pair: Vec<&str> = s.split(":").collect();
            let output_name_cstr = CString::new(output_name_pair[0]).expect("");
            let output_name_idx = if output_name_pair.len() > 1 {
                output_name_pair[1].parse().unwrap()
            } else {
                0
            };

            ssvm_tensorflow_append_output(
                output_name_cstr.as_ptr() as *const u8,
                output_name_cstr.as_bytes().len() as u32,
                output_name_idx,
            );
        }
        ssvm_tensorflow_run_vision(
            model_buf.as_ptr() as *const u8,
            model_buf.len() as u32,
            tensor_buf.as_ptr() as *const u8,
            (tensor_buf.len() * mem::size_of::<f32>()) as u32,
            image_width,
            image_height,
        );

        ssvm_tensorflow_clear_input();
        ssvm_tensorflow_clear_output();

        let mut res_vec: Vec<Vec<u8>> = Vec::new();

        for i in 0..output_names.len() as u32 {
            let result_len = ssvm_tensorflow_get_result_len(i);
            let mut result_vec: Vec<u8> = vec![0; result_len as usize];
            let result_ptr = result_vec.as_mut_ptr();
            ssvm_tensorflow_get_result(i, result_ptr);
            res_vec.push(result_vec);
        }

        Tensors { data: res_vec }
    }
}
