"""
加密解密工具模块

提供完整的加密解密功能，包括：
- 对称加密（AES、DES、3DES）
- 非对称加密（RSA）
- 哈希计算（MD5、SHA系列）
- HMAC 消息认证
- Base64 编码/解码
- 文件加密解密
- 密码哈希
"""

import os
import hashlib
import base64
import secrets
import hmac
from typing import Optional, Union, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


try:
    from cryptography.hazmat.primitives import hashes, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
    from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class HashAlgorithm(Enum):
    """哈希算法枚举"""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA224 = "sha224"
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


class CipherMode(Enum):
    """加密模式枚举"""
    CBC = "cbc"
    ECB = "ecb"
    CFB = "cfb"
    OFB = "ofb"
    CTR = "ctr"
    GCM = "gcm"


@dataclass
class KeyPair:
    """密钥对数据类"""
    public_key: Any
    private_key: Any
    public_key_pem: str
    private_key_pem: str


@dataclass
class EncryptedData:
    """加密数据数据类"""
    ciphertext: bytes
    iv: Optional[bytes]
    tag: Optional[bytes]
    salt: Optional[bytes]
    algorithm: str


class EncryptDecrypt:
    """
    加密解密工具类

    提供完整的加密解密功能
    """

    @staticmethod
    def is_available() -> bool:
        """检查加密库是否可用"""
        return CRYPTO_AVAILABLE

    @staticmethod
    def ensure_crypto():
        """确保加密库可用"""
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography 库未安装。请运行: pip install cryptography"
            )

    @staticmethod
    def generate_key(length: int = 32, hex: bool = False) -> Union[bytes, str]:
        """
        生成随机密钥

        Args:
            length: 密钥长度（字节）
            hex: 是否返回十六进制字符串

        Returns:
            密钥（bytes 或 str）
        """
        key = secrets.token_bytes(length)
        if hex:
            return key.hex()
        return key

    @staticmethod
    def generate_salt(length: int = 16) -> bytes:
        """
        生成随机盐值

        Args:
            length: 盐值长度（字节）

        Returns:
            盐值
        """
        return secrets.token_bytes(length)

    @staticmethod
    def generate_nonce(length: int = 16) -> bytes:
        """
        生成随机 nonce（初始化向量）

        Args:
            length: nonce 长度（字节）

        Returns:
            nonce
        """
        return secrets.token_bytes(length)

    @staticmethod
    def hash_data(
        data: Union[str, bytes],
        algorithm: Union[str, HashAlgorithm] = HashAlgorithm.SHA256
    ) -> str:
        """
        计算数据的哈希值

        Args:
            data: 输入数据
            algorithm: 哈希算法

        Returns:
            十六进制哈希值

        Example:
            >>> EncryptDecrypt.hash_data("hello", HashAlgorithm.SHA256)
            '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
        """
        if isinstance(data, str):
            data = data.encode('utf-8')

        if isinstance(algorithm, str):
            algorithm = HashAlgorithm(algorithm.lower())

        hash_func = hashlib.new(algorithm.value)
        hash_func.update(data)
        return hash_func.hexdigest()

    @staticmethod
    def hash_file(
        filepath: str,
        algorithm: Union[str, HashAlgorithm] = HashAlgorithm.SHA256,
        chunk_size: int = 8192
    ) -> str:
        """
        计算文件的哈希值

        Args:
            filepath: 文件路径
            algorithm: 哈希算法
            chunk_size: 读取块大小

        Returns:
            十六进制哈希值
        """
        if isinstance(algorithm, str):
            algorithm = HashAlgorithm(algorithm.lower())

        hash_func = hashlib.new(algorithm.value)

        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_func.update(chunk)

        return hash_func.hexdigest()

    @staticmethod
    def hmac_sign(
        data: Union[str, bytes],
        key: Union[str, bytes],
        algorithm: Union[str, HashAlgorithm] = HashAlgorithm.SHA256
    ) -> str:
        """
        生成 HMAC 签名

        Args:
            data: 输入数据
            key: 密钥
            algorithm: 哈希算法

        Returns:
            十六进制 HMAC 值
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')

        if isinstance(algorithm, str):
            algorithm = HashAlgorithm(algorithm.lower())

        mac = hmac.new(key, data, hashlib.new(algorithm.value))
        return mac.hexdigest()

    @staticmethod
    def hmac_verify(
        data: Union[str, bytes],
        key: Union[str, bytes],
        signature: str,
        algorithm: Union[str, HashAlgorithm] = HashAlgorithm.SHA256
    ) -> bool:
        """
        验证 HMAC 签名

        Args:
            data: 输入数据
            key: 密钥
            signature: 要验证的签名
            algorithm: 哈希算法

        Returns:
            是否验证通过
        """
        expected = EncryptDecrypt.hmac_sign(data, key, algorithm)
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def aes_encrypt(
        plaintext: Union[str, bytes],
        key: Union[str, bytes],
        mode: Union[str, CipherMode] = CipherMode.CBC,
        iv: Optional[bytes] = None,
        use_padding: bool = True
    ) -> EncryptedData:
        """
        AES 加密

        Args:
            plaintext: 明文
            key: 密钥（16、24 或 32 字节）
            mode: 加密模式
            iv: 初始化向量（ CBC/CFB/OFB 模式需要）
            use_padding: 是否填充

        Returns:
            EncryptedData 对象

        Raises:
            ValueError: 参数无效
        """
        EncryptDecrypt.ensure_crypto()

        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        if isinstance(mode, str):
            mode = CipherMode(mode.lower())

        if len(key) not in [16, 24, 32]:
            raise ValueError("AES 密钥长度必须是 16、24 或 32 字节")

        mode_obj = None
        iv_used = None

        if mode == CipherMode.CBC:
            iv_used = iv or secrets.token_bytes(16)
            mode_obj = modes.CBC(iv_used)
        elif mode == CipherMode.ECB:
            mode_obj = modes.ECB()
        elif mode == CipherMode.CFB:
            iv_used = iv or secrets.token_bytes(16)
            mode_obj = modes.CFB(iv_used)
        elif mode == CipherMode.OFB:
            iv_used = iv or secrets.token_bytes(16)
            mode_obj = modes.OFB(iv_used)
        elif mode == CipherMode.CTR:
            iv_used = iv or secrets.token_bytes(16)
            mode_obj = modes.CTR(iv_used)
        elif mode == CipherMode.GCM:
            iv_used = iv or secrets.token_bytes(16)
            mode_obj = modes.GCM(iv_used)
        else:
            raise ValueError(f"不支持的加密模式: {mode}")

        cipher = Cipher(
            algorithms.AES(key),
            mode_obj,
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        if use_padding:
            padder = padding.PKCS7(128).padder()
            plaintext = padder.update(plaintext) + padder.finalize()

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        tag = None
        if mode == CipherMode.GCM:
            tag = encryptor.tag

        return EncryptedData(
            ciphertext=ciphertext,
            iv=iv_used,
            tag=tag,
            salt=None,
            algorithm=f"AES-{mode.value}"
        )

    @staticmethod
    def aes_decrypt(
        encrypted_data: EncryptedData,
        key: Union[str, bytes],
        use_padding: bool = True
    ) -> bytes:
        """
        AES 解密

        Args:
            encrypted_data: EncryptedData 对象
            key: 密钥
            use_padding: 是否去填充

        Returns:
            解密后的明文

        Raises:
            ValueError: 参数无效
        """
        EncryptDecrypt.ensure_crypto()

        if isinstance(key, str):
            key = key.encode('utf-8')

        if len(key) not in [16, 24, 32]:
            raise ValueError("AES 密钥长度必须是 16、24 或 32 字节")

        ciphertext = encrypted_data.ciphertext
        iv = encrypted_data.iv

        mode_str = encrypted_data.algorithm.split('-')[1].lower()
        mode = CipherMode(mode_str)

        mode_obj = None
        if mode == CipherMode.CBC:
            mode_obj = modes.CBC(iv)
        elif mode == CipherMode.ECB:
            mode_obj = modes.ECB()
        elif mode == CipherMode.CFB:
            mode_obj = modes.CFB(iv)
        elif mode == CipherMode.OFB:
            mode_obj = modes.OFB(iv)
        elif mode == CipherMode.CTR:
            mode_obj = modes.CTR(iv)
        elif mode == CipherMode.GCM:
            mode_obj = modes.GCM(iv, encrypted_data.tag)
        else:
            raise ValueError(f"不支持的加密模式: {mode}")

        cipher = Cipher(
            algorithms.AES(key),
            mode_obj,
            backend=default_backend()
        )
        decryptor = cipher.decryptor()

        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        if use_padding:
            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(plaintext) + unpadder.finalize()

        return plaintext

    @staticmethod
    def encrypt_password(
        password: str,
        salt: Optional[bytes] = None,
        iterations: int = 100000,
        key_length: int = 32
    ) -> Tuple[str, str]:
        """
        加密密码（使用 PBKDF2）

        Args:
            password: 密码
            salt: 盐值，None 表示自动生成
            iterations: 迭代次数
            key_length: 输出密钥长度

        Returns:
            (加密结果十六进制字符串, 盐值十六进制字符串)
        """
        if salt is None:
            salt = EncryptDecrypt.generate_salt()

        password_bytes = password.encode('utf-8')
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password_bytes,
            salt,
            iterations,
            dklen=key_length
        )

        return key.hex(), salt.hex()

    @staticmethod
    def verify_password(
        password: str,
        encrypted: str,
        salt: str,
        iterations: int = 100000
    ) -> bool:
        """
        验证密码

        Args:
            password: 要验证的密码
            encrypted: 加密后的密码
            salt: 盐值（十六进制字符串）
            iterations: 迭代次数

        Returns:
            是否验证通过
        """
        salt_bytes = bytes.fromhex(salt)
        expected, _ = EncryptDecrypt.encrypt_password(
            password, salt_bytes, iterations
        )
        return hmac.compare_digest(expected, encrypted)

    @staticmethod
    def generate_rsa_keypair(
        key_size: int = 2048,
        public_exponent: int = 65537
    ) -> KeyPair:
        """
        生成 RSA 密钥对

        Args:
            key_size: 密钥大小（位数）
            public_exponent: 公开指数

        Returns:
            KeyPair 对象
        """
        EncryptDecrypt.ensure_crypto()

        private_key = generate_private_key(
            public_exponent=public_exponent,
            key_size=key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return KeyPair(
            public_key=public_key,
            private_key=private_key,
            public_key_pem=public_pem.decode('utf-8'),
            private_key_pem=private_pem.decode('utf-8')
        )

    @staticmethod
    def rsa_encrypt(
        plaintext: Union[str, bytes],
        public_key_pem: str
    ) -> bytes:
        """
        RSA 加密

        Args:
            plaintext: 明文
            public_key_pem: 公钥（PEM 格式）

        Returns:
            密文
        """
        EncryptDecrypt.ensure_crypto()

        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')

        public_key = serialization.load_pem_public_key(
            public_key_pem.encode('utf-8'),
            backend=default_backend()
        )

        ciphertext = public_key.encrypt(
            plaintext,
            asym_padding.PKCS1v15()
        )

        return ciphertext

    @staticmethod
    def rsa_decrypt(
        ciphertext: bytes,
        private_key_pem: str
    ) -> bytes:
        """
        RSA 解密

        Args:
            ciphertext: 密文
            private_key_pem: 私钥（PEM 格式）

        Returns:
            明文
        """
        EncryptDecrypt.ensure_crypto()

        private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None,
            backend=default_backend()
        )

        plaintext = private_key.decrypt(
            ciphertext,
            asym_padding.PKCS1v15()
        )

        return plaintext

    @staticmethod
    def rsa_sign(
        data: Union[str, bytes],
        private_key_pem: str,
        algorithm: Union[str, HashAlgorithm] = HashAlgorithm.SHA256
    ) -> bytes:
        """
        RSA 签名

        Args:
            data: 要签名的数据
            private_key_pem: 私钥（PEM 格式）
            algorithm: 哈希算法

        Returns:
            签名
        """
        EncryptDecrypt.ensure_crypto()

        if isinstance(data, str):
            data = data.encode('utf-8')

        if isinstance(algorithm, str):
            algorithm = HashAlgorithm(algorithm.lower())

        hash_algorithm = getattr(hashes, algorithm.name)()

        private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None,
            backend=default_backend()
        )

        signature = private_key.sign(
            data,
            asym_padding.PKCS1v15(),
            hash_algorithm
        )

        return signature

    @staticmethod
    def rsa_verify(
        data: Union[str, bytes],
        signature: bytes,
        public_key_pem: str,
        algorithm: Union[str, HashAlgorithm] = HashAlgorithm.SHA256
    ) -> bool:
        """
        RSA 验签

        Args:
            data: 原始数据
            signature: 签名
            public_key_pem: 公钥（PEM 格式）
            algorithm: 哈希算法

        Returns:
            是否验证通过
        """
        EncryptDecrypt.ensure_crypto()

        if isinstance(data, str):
            data = data.encode('utf-8')

        if isinstance(algorithm, str):
            algorithm = HashAlgorithm(algorithm.lower())

        hash_algorithm = getattr(hashes, algorithm.name)()

        public_key = serialization.load_pem_public_key(
            public_key_pem.encode('utf-8'),
            backend=default_backend()
        )

        try:
            public_key.verify(
                signature,
                data,
                asym_padding.PKCS1v15(),
                hash_algorithm
            )
            return True
        except Exception:
            return False

    @staticmethod
    def encrypt_file(
        input_path: str,
        output_path: str,
        key: Union[str, bytes],
        mode: Union[str, CipherMode] = CipherMode.CBC,
        chunk_size: int = 1024 * 1024
    ) -> str:
        """
        加密文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            key: 密钥
            mode: 加密模式
            chunk_size: 块大小

        Returns:
            输出文件路径
        """
        if isinstance(key, str):
            key = key.encode('utf-8')

        mode_obj = CipherMode(mode.lower()) if isinstance(mode, str) else mode

        encrypted = EncryptDecrypt.aes_encrypt(
            b"",
            key,
            mode_obj
        )
        iv = encrypted.iv

        with open(input_path, 'rb') as fin:
            with open(output_path, 'wb') as fout:
                fout.write(iv)

                while True:
                    chunk = fin.read(chunk_size)
                    if not chunk:
                        break

                    encrypted = EncryptDecrypt.aes_encrypt(
                        chunk,
                        key,
                        mode_obj,
                        iv
                    )
                    fout.write(len(encrypted.ciphertext).to_bytes(4, 'big'))
                    fout.write(encrypted.ciphertext)
                    iv = encrypted.iv

        return output_path

    @staticmethod
    def decrypt_file(
        input_path: str,
        output_path: str,
        key: Union[str, bytes]
    ) -> str:
        """
        解密文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            key: 密钥

        Returns:
            输出文件路径
        """
        if isinstance(key, str):
            key = key.encode('utf-8')

        with open(input_path, 'rb') as fin:
            iv = fin.read(16)

            with open(output_path, 'wb') as fout:
                previous_ct = iv

                while True:
                    length_bytes = fin.read(4)
                    if not length_bytes:
                        break

                    length = int.from_bytes(length_bytes, 'big')
                    ciphertext = fin.read(length)

                    encrypted = EncryptedData(
                        ciphertext=ciphertext,
                        iv=iv,
                        tag=None,
                        salt=None,
                        algorithm="AES-cbc"
                    )

                    plaintext = EncryptDecrypt.aes_decrypt(
                        encrypted,
                        key,
                        padding=False
                    )

                    fout.write(plaintext)
                    previous_ct = ciphertext
                    iv = previous_ct[:16]

        return output_path


def encrypt(plaintext: Union[str, bytes], key: Union[str, bytes]) -> str:
    """
    便捷函数：加密数据

    Args:
        plaintext: 明文
        key: 密钥

    Returns:
        Base64 编码的密文
    """
    if isinstance(key, str):
        key_bytes = key.encode('utf-8')
    else:
        key_bytes = key

    if len(key_bytes) < 16:
        key_bytes = key_bytes + b'\0' * (16 - len(key_bytes))
    elif len(key_bytes) > 32:
        key_bytes = key_bytes[:32]
    elif len(key_bytes) not in (16, 24, 32):
        key_bytes = key_bytes.ljust(32, b'\0')[:32]

    encrypted = EncryptDecrypt.aes_encrypt(plaintext, key_bytes)
    combined = (encrypted.iv or b'') + encrypted.ciphertext
    return base64.b64encode(combined).decode('utf-8')


def decrypt(ciphertext: Union[str, bytes], key: Union[str, bytes]) -> str:
    """
    便捷函数：解密数据

    Args:
        ciphertext: Base64 编码的密文
        key: 密钥

    Returns:
        解密后的明文
    """
    if isinstance(ciphertext, str):
        ciphertext = base64.b64decode(ciphertext)

    if isinstance(key, str):
        key_bytes = key.encode('utf-8')
    else:
        key_bytes = key

    if len(key_bytes) < 16:
        key_bytes = key_bytes + b'\0' * (16 - len(key_bytes))
    elif len(key_bytes) > 32:
        key_bytes = key_bytes[:32]
    elif len(key_bytes) not in (16, 24, 32):
        key_bytes = key_bytes.ljust(32, b'\0')[:32]

    iv = ciphertext[:16]
    data = ciphertext[16:]

    encrypted = EncryptedData(
        ciphertext=data,
        iv=iv,
        tag=None,
        salt=None,
        algorithm="AES-cbc"
    )

    plaintext = EncryptDecrypt.aes_decrypt(encrypted, key_bytes)
    return plaintext.decode('utf-8')


def hash_string(data: str, algorithm: str = "sha256") -> str:
    """
    便捷函数：计算字符串哈希

    Args:
        data: 输入字符串
        algorithm: 哈希算法

    Returns:
        十六进制哈希值
    """
    return EncryptDecrypt.hash_data(data, algorithm)


def hash_bytes(data: bytes, algorithm: str = "sha256") -> str:
    """
    便捷函数：计算字节串哈希

    Args:
        data: 输入字节串
        algorithm: 哈希算法

    Returns:
        十六进制哈希值
    """
    return EncryptDecrypt.hash_data(data, algorithm)


def generate_password(length: int = 16, include_special: bool = True) -> str:
    """
    便捷函数：生成随机密码

    Args:
        length: 密码长度
        include_special: 是否包含特殊字符

    Returns:
        随机密码
    """
    import string

    chars = string.ascii_letters + string.digits
    if include_special:
        chars += string.punctuation

    password = ''.join(secrets.choice(chars) for _ in range(length))
    return password


def secure_compare(a: Union[str, bytes], b: Union[str, bytes]) -> bool:
    """
    便捷函数：安全比较（防止时序攻击）

    Args:
        a: 第一个值
        b: 第二个值

    Returns:
        是否相等
    """
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')
    return hmac.compare_digest(a, b)


def encode_base64(data: Union[str, bytes]) -> str:
    """
    便捷函数：Base64 编码

    Args:
        data: 输入数据

    Returns:
        Base64 编码字符串
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')


def decode_base64(data: str) -> bytes:
    """
    便捷函数：Base64 解码

    Args:
        data: Base64 编码字符串

    Returns:
        解码后的字节串
    """
    return base64.b64decode(data)


def encode_url_safe(data: Union[str, bytes]) -> str:
    """
    便捷函数：URL 安全 Base64 编码

    Args:
        data: 输入数据

    Returns:
        URL 安全 Base64 编码字符串
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')


def decode_url_safe(data: str) -> bytes:
    """
    便捷函数：URL 安全 Base64 解码

    Args:
        data: URL 安全 Base64 编码字符串

    Returns:
        解码后的字节串
    """
    padding = 4 - len(data) % 4
    if padding != 4:
        data += '=' * padding
    return base64.urlsafe_b64decode(data)
