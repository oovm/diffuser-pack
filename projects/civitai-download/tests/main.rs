use curl::easy::Easy;

use civitai::{RequestAllModels, RequestModel};
use std::io::Read;

#[test]
fn ready() {
    println!("it works!")
}

#[tokio::test]
async fn test_all() {
    let models = RequestAllModels::default().send().await.unwrap();
    println!("{:#?}", models.next_page());
}

#[tokio::test]
async fn test_model1() {
    let models = RequestModel::new(1).send().await.unwrap();
    println!("{:#?}", models);
}

#[test]
fn main() {
    let mut data = "this is the body".as_bytes();

    let mut easy = Easy::new();
    easy.url("http://www.example.com/upload").unwrap();
    easy.post(true).unwrap();
    easy.post_field_size(data.len() as u64).unwrap();

    let mut transfer = easy.transfer();
    transfer.read_function(|buf| Ok(data.read(buf).unwrap_or(0))).unwrap();
    transfer.perform().unwrap();
}
