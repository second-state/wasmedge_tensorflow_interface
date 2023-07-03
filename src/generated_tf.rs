// This file is automatically generated, DO NOT EDIT
//
// To regenerate this file run the `crates/witx-bindgen` command

use core::fmt;
use core::mem::MaybeUninit;
#[repr(transparent)]
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct WasmedgeTfErrno(u32);
pub const WASMEDGE_TF_ERRNO_SUCCESS: WasmedgeTfErrno = WasmedgeTfErrno(0);
pub const WASMEDGE_TF_ERRNO_INVALID_ARGUMENT: WasmedgeTfErrno = WasmedgeTfErrno(1);
pub const WASMEDGE_TF_ERRNO_INVALID_ENCODING: WasmedgeTfErrno = WasmedgeTfErrno(2);
pub const WASMEDGE_TF_ERRNO_MISSING_MEMORY: WasmedgeTfErrno = WasmedgeTfErrno(3);
pub const WASMEDGE_TF_ERRNO_BUSY: WasmedgeTfErrno = WasmedgeTfErrno(4);
pub const WASMEDGE_TF_ERRNO_RUNTIME_ERROR: WasmedgeTfErrno = WasmedgeTfErrno(5);
impl WasmedgeTfErrno {
    pub const fn raw(&self) -> u32 {
        self.0
    }

    pub fn name(&self) -> &'static str {
        match self.0 {
            0 => "SUCCESS",
            1 => "INVALID_ARGUMENT",
            2 => "INVALID_ENCODING",
            3 => "MISSING_MEMORY",
            4 => "BUSY",
            5 => "RUNTIME_ERROR",
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }
    pub fn message(&self) -> &'static str {
        match self.0 {
            0 => "",
            1 => "",
            2 => "",
            3 => "",
            4 => "",
            5 => "",
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }
}
impl fmt::Debug for WasmedgeTfErrno {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WasmedgeTfErrno")
            .field("code", &self.0)
            .field("name", &self.name())
            .field("message", &self.message())
            .finish()
    }
}
impl fmt::Display for WasmedgeTfErrno {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (error {})", self.name(), self.0)
    }
}

#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
impl std::error::Error for WasmedgeTfErrno {}

pub type ModelBuffer<'a> = &'a [u8];
pub type MetagraphTag<'a> = &'a str;
pub type MetagraphTagList<'a> = &'a [MetagraphTag<'a>];
pub type Session = u32;
pub type Tensor = u32;
pub type TensorSize = u32;
pub type TensorType = u32;
pub type TensorData<'a> = &'a [u8];
pub type TensorDimensions<'a> = &'a [u64];
pub unsafe fn create_session(model_buffer: ModelBuffer<'_>) -> Result<Session, WasmedgeTfErrno> {
    let mut rp0 = MaybeUninit::<Session>::uninit();
    let ret = wasmedge_tensorflow::create_session(
        model_buffer.as_ptr() as i32,
        model_buffer.len() as i32,
        rp0.as_mut_ptr() as i32,
    );
    match ret {
        0 => Ok(core::ptr::read(rp0.as_mut_ptr() as i32 as *const Session)),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub unsafe fn create_session_saved_model(
    folder_path: &str,
    metagraph_tags: MetagraphTagList<'_>,
) -> Result<Session, WasmedgeTfErrno> {
    let mut rp0 = MaybeUninit::<Session>::uninit();
    let ret = wasmedge_tensorflow::create_session_saved_model(
        folder_path.as_ptr() as i32,
        folder_path.len() as i32,
        metagraph_tags.as_ptr() as i32,
        metagraph_tags.len() as i32,
        rp0.as_mut_ptr() as i32,
    );
    match ret {
        0 => Ok(core::ptr::read(rp0.as_mut_ptr() as i32 as *const Session)),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub unsafe fn delete_session(session: Session) -> Result<(), WasmedgeTfErrno> {
    let ret = wasmedge_tensorflow::delete_session(session as i32);
    match ret {
        0 => Ok(()),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub unsafe fn run_session(session: Session) -> Result<(), WasmedgeTfErrno> {
    let ret = wasmedge_tensorflow::run_session(session as i32);
    match ret {
        0 => Ok(()),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub unsafe fn get_output_tensor(session: Session, name: &str) -> Result<Tensor, WasmedgeTfErrno> {
    let mut rp0 = MaybeUninit::<Tensor>::uninit();
    let ret = wasmedge_tensorflow::get_output_tensor(
        session as i32,
        name.as_ptr() as i32,
        name.len() as i32,
        rp0.as_mut_ptr() as i32,
    );
    match ret {
        0 => Ok(core::ptr::read(rp0.as_mut_ptr() as i32 as *const Tensor)),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub unsafe fn get_tensor_len(
    session: Session,
    tensor: Tensor,
) -> Result<TensorSize, WasmedgeTfErrno> {
    let mut rp0 = MaybeUninit::<TensorSize>::uninit();
    let ret =
        wasmedge_tensorflow::get_tensor_len(session as i32, tensor as i32, rp0.as_mut_ptr() as i32);
    match ret {
        0 => Ok(core::ptr::read(rp0.as_mut_ptr() as i32 as *const TensorSize)),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub unsafe fn get_tensor_data(
    session: Session,
    tensor: Tensor,
    tensor_buf: *mut u8,
    tensor_buf_max_size: TensorSize,
) -> Result<TensorSize, WasmedgeTfErrno> {
    let mut rp0 = MaybeUninit::<TensorSize>::uninit();
    let ret = wasmedge_tensorflow::get_tensor_data(
        session as i32,
        tensor as i32,
        tensor_buf as i32,
        tensor_buf_max_size as i32,
        rp0.as_mut_ptr() as i32,
    );
    match ret {
        0 => Ok(core::ptr::read(rp0.as_mut_ptr() as i32 as *const TensorSize)),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub unsafe fn append_input(
    session: Session,
    name: &str,
    dimension: TensorDimensions<'_>,
    data_type: TensorType,
    tensor_buf: TensorData<'_>,
) -> Result<(), WasmedgeTfErrno> {
    let ret = wasmedge_tensorflow::append_input(
        session as i32,
        name.as_ptr() as i32,
        name.len() as i32,
        dimension.as_ptr() as i32,
        dimension.len() as i32,
        data_type as i32,
        tensor_buf.as_ptr() as i32,
        tensor_buf.len() as i32,
    );
    match ret {
        0 => Ok(()),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub unsafe fn append_output(session: Session, name: &str) -> Result<(), WasmedgeTfErrno> {
    let ret =
        wasmedge_tensorflow::append_output(session as i32, name.as_ptr() as i32, name.len() as i32);
    match ret {
        0 => Ok(()),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub unsafe fn clear_input(session: Session) -> Result<(), WasmedgeTfErrno> {
    let ret = wasmedge_tensorflow::clear_input(session as i32);
    match ret {
        0 => Ok(()),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub unsafe fn clear_output(session: Session) -> Result<(), WasmedgeTfErrno> {
    let ret = wasmedge_tensorflow::clear_output(session as i32);
    match ret {
        0 => Ok(()),
        _ => Err(WasmedgeTfErrno(ret as u32)),
    }
}

pub mod wasmedge_tensorflow {
    #[link(wasm_import_module = "wasmedge_tensorflow")]
    extern "C" {
        pub fn create_session(arg0: i32, arg1: i32, arg2: i32) -> i32;
        pub fn create_session_saved_model(
            arg0: i32,
            arg1: i32,
            arg2: i32,
            arg3: i32,
            arg4: i32,
        ) -> i32;
        pub fn delete_session(arg0: i32) -> i32;
        pub fn run_session(arg0: i32) -> i32;
        pub fn get_output_tensor(arg0: i32, arg1: i32, arg2: i32, arg3: i32) -> i32;
        pub fn get_tensor_len(arg0: i32, arg1: i32, arg2: i32) -> i32;
        pub fn get_tensor_data(arg0: i32, arg1: i32, arg2: i32, arg3: i32, arg4: i32) -> i32;
        pub fn append_input(
            arg0: i32,
            arg1: i32,
            arg2: i32,
            arg3: i32,
            arg4: i32,
            arg5: i32,
            arg6: i32,
            arg7: i32,
        ) -> i32;
        pub fn append_output(arg0: i32, arg1: i32, arg2: i32) -> i32;
        pub fn clear_input(arg0: i32) -> i32;
        pub fn clear_output(arg0: i32) -> i32;
    }
}