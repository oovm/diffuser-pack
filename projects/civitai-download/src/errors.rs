
#[test]
fn main() {
    let mut data = "this is the body".as_bytes();

    let mut easy = Easy::new();
    easy.url("http://www.example.com/upload").unwrap();
    easy.post(true).unwrap();
    easy.post_field_size(data.len() as u64).unwrap();

    let mut transfer = easy.transfer();
    transfer.read_function(|buf| {
        Ok(data.read(buf).unwrap_or(0))
    }).unwrap();
    transfer.perform().unwrap();
}