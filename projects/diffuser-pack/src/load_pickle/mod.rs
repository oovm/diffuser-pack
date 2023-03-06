use std::fs::{File};
use safetensors::tensor::{SafeTensorError, SafeTensors};
use serde_pickle::{DeOptions, value_from_reader};
use std::path::Path;


#[test]
fn test_load() {
    let path = "pureerosface_v1.pt";
    load_pth(&Path::new(path)).unwrap();
}


pub fn load_pth(path: &Path) -> Result<SafeTensors, SafeTensorError> {
    let mut zip = zip::ZipArchive::new(File::open(path).unwrap()).unwrap();
    for i in 0..zip.len() {
        let mut file = zip.by_index(i).unwrap();
        if file.name().eq_ignore_ascii_case("archive/data.pkl") {
            let config = DeOptions::default().replace_unresolved_globals();
            let value = match value_from_reader(&mut file, config) {
                Ok(o) => o,
                Err(e) => Err(SafeTensorError::InvalidOffset(e.to_string()))?,
            };
            println!("{:#?}", value)
        }
        else if file.name().starts_with("archive/data/") {
            // collect tensors
        }
    }
    todo!()
}