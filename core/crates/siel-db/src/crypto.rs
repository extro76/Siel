use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng},
    ChaCha20Poly1305, Key, Nonce,
};
use siel_core::{SielError, Result};
use rand::RngCore;
use sha2::{Digest, Sha256};

pub struct EncryptedPayload {
    pub question_cipher: Vec<u8>,
    pub answer_cipher: Vec<u8>,
    pub nonce: Vec<u8>,
    pub aad: Vec<u8>,
}

#[derive(Clone)]
pub struct CryptoBox {
    master_key: [u8; 32],
}

impl CryptoBox {
    pub fn new(master_secret: &[u8]) -> Self {
        let digest = Sha256::digest(master_secret);
        let mut master_key = [0_u8; 32];
        master_key.copy_from_slice(&digest);
        Self { master_key }
    }

    pub fn generate_item_key(&self) -> [u8; 32] {
        let mut key = [0_u8; 32];
        OsRng.fill_bytes(&mut key);
        key
    }

    pub fn wrap_key(&self, item_key: &[u8; 32]) -> Vec<u8> {
        xor32(item_key, &self.master_key).to_vec()
    }

    pub fn unwrap_key(&self, wrapped_key: &[u8]) -> Result<[u8; 32]> {
        if wrapped_key.len() != 32 {
            return Err(SielError::Crypto("wrapped key is invalid or destroyed".to_string()));
        }
        let mut wrapped = [0_u8; 32];
        wrapped.copy_from_slice(wrapped_key);
        Ok(xor32(&wrapped, &self.master_key))
    }

    pub fn encrypt_payload(
        &self,
        item_key: &[u8; 32],
        question: &str,
        answer: &str,
        aad: &[u8],
    ) -> Result<EncryptedPayload> {
        let cipher = ChaCha20Poly1305::new(Key::from_slice(item_key));
        let mut question_nonce = [0_u8; 12];
        let mut answer_nonce = [0_u8; 12];
        OsRng.fill_bytes(&mut question_nonce);
        OsRng.fill_bytes(&mut answer_nonce);
        let question_cipher = cipher
            .encrypt(
                Nonce::from_slice(&question_nonce),
                chacha20poly1305::aead::Payload {
                    msg: question.as_bytes(),
                    aad,
                },
            )
            .map_err(|_| SielError::Crypto("question encryption failed".to_string()))?;
        let answer_cipher = cipher
            .encrypt(
                Nonce::from_slice(&answer_nonce),
                chacha20poly1305::aead::Payload {
                    msg: answer.as_bytes(),
                    aad,
                },
            )
            .map_err(|_| SielError::Crypto("answer encryption failed".to_string()))?;
        let mut nonce = Vec::with_capacity(24);
        nonce.extend_from_slice(&question_nonce);
        nonce.extend_from_slice(&answer_nonce);
        Ok(EncryptedPayload {
            question_cipher,
            answer_cipher,
            nonce,
            aad: aad.to_vec(),
        })
    }

    pub fn decrypt_payload(
        &self,
        item_key: &[u8; 32],
        question_cipher: &[u8],
        answer_cipher: &[u8],
        nonce: &[u8],
        aad: &[u8],
    ) -> Result<(String, String)> {
        if nonce.len() != 24 {
            return Err(SielError::Crypto("nonce length is invalid".to_string()));
        }
        let (question_nonce, answer_nonce) = nonce.split_at(12);
        let cipher = ChaCha20Poly1305::new(Key::from_slice(item_key));
        let question = cipher
            .decrypt(
                Nonce::from_slice(question_nonce),
                chacha20poly1305::aead::Payload {
                    msg: question_cipher,
                    aad,
                },
            )
            .map_err(|_| SielError::Crypto("question decryption failed".to_string()))?;
        let answer = cipher
            .decrypt(
                Nonce::from_slice(answer_nonce),
                chacha20poly1305::aead::Payload {
                    msg: answer_cipher,
                    aad,
                },
            )
            .map_err(|_| SielError::Crypto("answer decryption failed".to_string()))?;
        Ok((
            String::from_utf8(question)
                .map_err(|err| SielError::Crypto(format!("question is not UTF-8: {err}")))?,
            String::from_utf8(answer)
                .map_err(|err| SielError::Crypto(format!("answer is not UTF-8: {err}")))?,
        ))
    }
}

fn xor32(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut out = [0_u8; 32];
    for idx in 0..32 {
        out[idx] = a[idx] ^ b[idx];
    }
    out
}
