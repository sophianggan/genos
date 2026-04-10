//! GGUF v3 parser for `no_std` environments.
//!
//! Parses quantized model files in the GGUF format, providing access to
//! metadata key-value pairs and tensor data. Supports GGUF v2 and v3.
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

use alloc::string::String;
use alloc::vec::Vec;

// ── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGUFError {
    InvalidMagic,
    UnsupportedVersion,
    Truncated,
    InvalidValueType,
    InvalidTensorType,
    TensorNotFound,
    BadAlignment,
}

// ── GGML quantisation types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
}

impl GGMLType {
    pub fn from_u32(v: u32) -> Result<Self, GGUFError> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            30 => Ok(Self::BF16),
            _ => Err(GGUFError::InvalidTensorType),
        }
    }

    /// Block size for quantised types, 1 for scalar types.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Bytes per block (or per element for scalar types).
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,   // 2 (f16 scale) + 16 (4-bit values)
            Self::Q4_1 => 20,   // 2 (f16 d) + 2 (f16 m) + 16 (4-bit values)
            Self::Q5_0 => 22,   // 2 (f16) + 4 (high bits) + 16 (low nibbles)
            Self::Q5_1 => 24,   // 2 + 2 + 4 + 16
            Self::Q8_0 => 34,   // 2 (f16 scale) + 32 (i8 values)
            Self::Q8_1 => 40,   // 2 (f16 d) + 2 (f16 s) + 4 (padding) + 32
            Self::Q2K => 84,    // super-block of 256
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
        }
    }

    /// Compute the byte size of a tensor with `n_elements` values.
    pub fn tensor_bytes(self, n_elements: u64) -> u64 {
        let bs = self.block_size() as u64;
        let bb = self.block_bytes() as u64;
        (n_elements / bs) * bb
    }
}

// ── Metadata values ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<GGUFValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GGUFValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            Self::U64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(*v as u64),
            Self::I64(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[GGUFValue]> {
        match self {
            Self::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

// ── Tensor metadata ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GGMLType,
    /// Offset from the start of the data section (NOT from file start).
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements in this tensor.
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().copied().product::<u64>().max(1)
    }

    /// Total bytes occupied by this tensor.
    pub fn byte_size(&self) -> u64 {
        self.dtype.tensor_bytes(self.n_elements())
    }
}

// ── Main GGUF file ───────────────────────────────────────────────────────────

/// Parsed GGUF file. Holds references into the original data buffer.
#[derive(Debug)]
pub struct GGUFFile<'a> {
    pub version: u32,
    pub metadata: Vec<(String, GGUFValue)>,
    pub tensors: Vec<TensorInfo>,
    /// Offset in `data` where tensor binary data begins.
    pub data_offset: usize,
    /// The full file data (kept as a reference for zero-copy tensor access).
    data: &'a [u8],
}

impl<'a> GGUFFile<'a> {
    /// Parse a GGUF file from a raw byte buffer.
    pub fn parse(data: &'a [u8]) -> Result<Self, GGUFError> {
        let mut r = Reader::new(data);

        // Magic
        let magic = r.u32()?;
        if magic != 0x46554747 {
            return Err(GGUFError::InvalidMagic);
        }

        // Version
        let version = r.u32()?;
        if version < 2 || version > 3 {
            return Err(GGUFError::UnsupportedVersion);
        }

        // Counts (v3 → u64, v2 → u32)
        let tensor_count;
        let metadata_kv_count;
        if version >= 3 {
            tensor_count = r.u64()?;
            metadata_kv_count = r.u64()?;
        } else {
            tensor_count = r.u32()? as u64;
            metadata_kv_count = r.u32()? as u64;
        }

        // Metadata
        let mut metadata = Vec::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = r.gguf_string()?;
            let vtype = r.u32()?;
            let value = r.gguf_value(vtype)?;
            metadata.push((key, value));
        }

        // Tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = r.gguf_string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let dtype = GGMLType::from_u32(r.u32()?)?;
            let offset = r.u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }

        // Data section alignment (default 32)
        let alignment = Self::find_alignment(&metadata);
        let data_offset = align_up(r.pos, alignment);

        Ok(GGUFFile {
            version,
            metadata,
            tensors,
            data_offset,
            data,
        })
    }

    // ── Metadata helpers ─────────────────────────────────────────────────

    /// Look up a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GGUFValue> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get_metadata(key).and_then(|v| v.as_str())
    }
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get_metadata(key).and_then(|v| v.as_u32())
    }
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get_metadata(key).and_then(|v| v.as_u64())
    }
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get_metadata(key).and_then(|v| v.as_f32())
    }
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get_metadata(key).and_then(|v| v.as_bool())
    }

    // ── Tensor access ────────────────────────────────────────────────────

    /// Find tensor info by name.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get the raw bytes for a tensor (returns a slice into the file buffer).
    pub fn tensor_data(&self, info: &TensorInfo) -> Result<&'a [u8], GGUFError> {
        let start = self.data_offset + info.offset as usize;
        let end = start + info.byte_size() as usize;
        if end > self.data.len() {
            return Err(GGUFError::Truncated);
        }
        Ok(&self.data[start..end])
    }

    /// Convenience: find a tensor and return its data.
    pub fn get_tensor_data(&self, name: &str) -> Result<(&TensorInfo, &'a [u8]), GGUFError> {
        let info = self.find_tensor(name).ok_or(GGUFError::TensorNotFound)?;
        let data = self.tensor_data(info)?;
        Ok((info, data))
    }

    // ── Private ──────────────────────────────────────────────────────────

    fn find_alignment(metadata: &[(String, GGUFValue)]) -> usize {
        for (k, v) in metadata {
            if k == "general.alignment" {
                if let Some(a) = v.as_u32() {
                    if a > 0 {
                        return a as usize;
                    }
                }
            }
        }
        32 // default alignment
    }
}

// ── Byte reader ──────────────────────────────────────────────────────────────

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], GGUFError> {
        if self.remaining() < n {
            return Err(GGUFError::Truncated);
        }
        let start = self.pos;
        self.pos += n;
        Ok(&self.buf[start..start + n])
    }

    fn u8(&mut self) -> Result<u8, GGUFError> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }
    fn i8(&mut self) -> Result<i8, GGUFError> {
        Ok(self.u8()? as i8)
    }
    fn u16(&mut self) -> Result<u16, GGUFError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn i16(&mut self) -> Result<i16, GGUFError> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32, GGUFError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32(&mut self) -> Result<i32, GGUFError> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, GGUFError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn i64(&mut self) -> Result<i64, GGUFError> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn f32(&mut self) -> Result<f32, GGUFError> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn f64(&mut self) -> Result<f64, GGUFError> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a GGUF string: u64 length + UTF-8 bytes.
    fn gguf_string(&mut self) -> Result<String, GGUFError> {
        let len = self.u64()? as usize;
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Read a GGUF metadata value of the given type tag.
    fn gguf_value(&mut self, vtype: u32) -> Result<GGUFValue, GGUFError> {
        match vtype {
            0 => Ok(GGUFValue::U8(self.u8()?)),
            1 => Ok(GGUFValue::I8(self.i8()?)),
            2 => Ok(GGUFValue::U16(self.u16()?)),
            3 => Ok(GGUFValue::I16(self.i16()?)),
            4 => Ok(GGUFValue::U32(self.u32()?)),
            5 => Ok(GGUFValue::I32(self.i32()?)),
            6 => Ok(GGUFValue::F32(self.f32()?)),
            7 => {
                let b = self.u8()?;
                Ok(GGUFValue::Bool(b != 0))
            }
            8 => {
                let s = self.gguf_string()?;
                Ok(GGUFValue::Str(s))
            }
            9 => {
                // Array: element_type (u32) + count (u64) + elements
                let elem_type = self.u32()?;
                let count = self.u64()? as usize;
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(self.gguf_value(elem_type)?);
                }
                Ok(GGUFValue::Array(arr))
            }
            10 => Ok(GGUFValue::U64(self.u64()?)),
            11 => Ok(GGUFValue::I64(self.i64()?)),
            12 => Ok(GGUFValue::F64(self.f64()?)),
            _ => Err(GGUFError::InvalidValueType),
        }
    }
}

// ── Alignment helper ─────────────────────────────────────────────────────────

fn align_up(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a minimal valid GGUF v3 file with one metadata entry and one tensor.
    fn build_test_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic "GGUF" little-endian
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count: 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // Metadata count: 1
        buf.extend_from_slice(&1u64.to_le_bytes());

        // Metadata: key = "general.architecture", value = string "gemma4"
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // type: string
        let val = b"gemma4";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        // Tensor: name = "test.weight", 1-D shape [64], type F32, offset 0
        let tname = b"test.weight";
        buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        buf.extend_from_slice(tname);
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&64u64.to_le_bytes()); // dim[0]
        buf.extend_from_slice(&0u32.to_le_bytes()); // type F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Align to 32 bytes
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        // Tensor data: 64 × f32, each = 1.0
        for _ in 0..64 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }

        buf
    }

    #[test]
    fn parse_gguf_header() {
        let data = build_test_gguf();
        let file = GGUFFile::parse(&data).unwrap();
        assert_eq!(file.version, 3);
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.metadata.len(), 1);
    }

    #[test]
    fn read_metadata() {
        let data = build_test_gguf();
        let file = GGUFFile::parse(&data).unwrap();
        assert_eq!(file.get_str("general.architecture"), Some("gemma4"));
    }

    #[test]
    fn read_tensor() {
        let data = build_test_gguf();
        let file = GGUFFile::parse(&data).unwrap();
        let (info, tdata) = file.get_tensor_data("test.weight").unwrap();
        assert_eq!(info.dtype, GGMLType::F32);
        assert_eq!(info.n_elements(), 64);
        assert_eq!(tdata.len(), 256); // 64 × 4 bytes
        // First element should be 1.0
        let first = f32::from_le_bytes([tdata[0], tdata[1], tdata[2], tdata[3]]);
        assert_eq!(first, 1.0);
    }

    #[test]
    fn invalid_magic() {
        let data = vec![0u8; 32];
        assert_eq!(GGUFFile::parse(&data).unwrap_err(), GGUFError::InvalidMagic);
    }

    #[test]
    fn tensor_not_found() {
        let data = build_test_gguf();
        let file = GGUFFile::parse(&data).unwrap();
        assert_eq!(
            file.get_tensor_data("nonexistent").unwrap_err(),
            GGUFError::TensorNotFound
        );
    }
}
