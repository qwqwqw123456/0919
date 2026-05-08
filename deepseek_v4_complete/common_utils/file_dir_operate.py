"""
文件和目录操作工具模块

提供完整的文件和目录操作功能，包括：
- 目录创建和管理
- 文件复制、移动、删除
- 文件搜索和过滤
- 文件路径操作
- 文件锁和并发控制
- 文件监控
- 压缩和解压
"""

import os
import sys
import shutil
import glob
import hashlib
import mimetypes
import filecmp
import tempfile
import tarfile
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from contextlib import contextmanager
from fnmatch import fnmatch


@dataclass
class FileInfo:
    """文件信息数据类"""
    path: str
    name: str
    size: int
    created_time: datetime
    modified_time: datetime
    is_directory: bool
    extension: str
    mime_type: Optional[str] = None


@dataclass
class DirectoryTree:
    """目录树结构"""
    path: str
    name: str
    is_directory: bool
    children: List['DirectoryTree'] = None
    size: Optional[int] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class FileDirOperate:
    """
    文件和目录操作工具类

    提供静态方法和便捷函数
    """

    @staticmethod
    def ensure_dir(path: str, mode: int = 0o755) -> str:
        """
        确保目录存在，不存在则创建

        Args:
            path: 目录路径
            mode: 目录权限

        Returns:
            目录路径
        """
        path = os.path.expanduser(path)
        os.makedirs(path, mode=mode, exist_ok=True)
        return path

    @staticmethod
    def ensure_parent_dir(filepath: str) -> str:
        """
        确保文件的父目录存在

        Args:
            filepath: 文件路径

        Returns:
            文件路径
        """
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        return filepath

    @staticmethod
    def copy_file(src: str, dst: str, overwrite: bool = False) -> str:
        """
        复制文件

        Args:
            src: 源文件路径
            dst: 目标文件路径
            overwrite: 是否覆盖已存在的文件

        Returns:
            目标文件路径

        Raises:
            FileExistsError: 目标文件已存在且 overwrite=False
        """
        src = os.path.expanduser(src)
        dst = os.path.expanduser(dst)

        if not os.path.exists(src):
            raise FileNotFoundError(f"源文件不存在: {src}")

        if os.path.exists(dst) and not overwrite:
            raise FileExistsError(f"目标文件已存在: {dst}")

        FileDirOperate.ensure_parent_dir(dst)
        shutil.copy2(src, dst)
        return dst

    @staticmethod
    def move_file(src: str, dst: str, overwrite: bool = False) -> str:
        """
        移动文件

        Args:
            src: 源文件路径
            dst: 目标文件路径
            overwrite: 是否覆盖已存在的文件

        Returns:
            目标文件路径
        """
        src = os.path.expanduser(src)
        dst = os.path.expanduser(dst)

        if not os.path.exists(src):
            raise FileNotFoundError(f"源文件不存在: {src}")

        if os.path.exists(dst) and not overwrite:
            raise FileExistsError(f"目标文件已存在: {dst}")

        FileDirOperate.ensure_parent_dir(dst)
        shutil.move(src, dst)
        return dst

    @staticmethod
    def delete_file(path: str, missing_ok: bool = False) -> bool:
        """
        删除文件

        Args:
            path: 文件路径
            missing_ok: 文件不存在时是否不报错

        Returns:
            是否成功删除
        """
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            if missing_ok:
                return False
            raise FileNotFoundError(f"文件不存在: {path}")

        os.remove(path)
        return True

    @staticmethod
    def delete_dir(path: str, recursive: bool = False, missing_ok: bool = False) -> bool:
        """
        删除目录

        Args:
            path: 目录路径
            recursive: 是否递归删除
            missing_ok: 目录不存在时是否不报错

        Returns:
            是否成功删除
        """
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            if missing_ok:
                return False
            raise FileNotFoundError(f"目录不存在: {path}")

        if recursive:
            shutil.rmtree(path)
        else:
            os.rmdir(path)
        return True

    @staticmethod
    def copy_dir(src: str, dst: str, symlinks: bool = False, 
                 ignore_patterns: Optional[List[str]] = None) -> str:
        """
        复制目录

        Args:
            src: 源目录路径
            dst: 目标目录路径
            symlinks: 是否保留符号链接
            ignore_patterns: 要忽略的文件模式列表

        Returns:
            目标目录路径
        """
        src = os.path.expanduser(src)
        dst = os.path.expanduser(dst)

        if not os.path.exists(src):
            raise FileNotFoundError(f"源目录不存在: {src}")

        if not os.path.isdir(src):
            raise NotADirectoryError(f"源路径不是目录: {src}")

        ignore_func = None
        if ignore_patterns:
            def ignore_func(path, names):
                ignored = set()
                for pattern in ignore_patterns:
                    ignored.update(fnmatch(names, pattern) for _ in names)
                return ignored

        shutil.copytree(src, dst, symlinks=symlinks, ignore=ignore_func)
        return dst

    @staticmethod
    def list_files(
        directory: str,
        pattern: str = "*",
        recursive: bool = False,
        include_dirs: bool = False
    ) -> List[str]:
        """
        列出目录中的文件

        Args:
            directory: 目录路径
            pattern: 文件名匹配模式
            recursive: 是否递归搜索
            include_dirs: 是否包含目录

        Returns:
            文件路径列表
        """
        directory = os.path.expanduser(directory)

        if not os.path.exists(directory):
            return []

        if recursive:
            results = []
            for root, dirs, files in os.walk(directory):
                if include_dirs:
                    for d in dirs:
                        full_path = os.path.join(root, d)
                        if fnmatch(d, pattern):
                            results.append(full_path)
                for f in files:
                    full_path = os.path.join(root, f)
                    if fnmatch(f, pattern):
                        results.append(full_path)
            return results
        else:
            results = []
            for item in os.listdir(directory):
                full_path = os.path.join(directory, item)
                if fnmatch(item, pattern):
                    if include_dirs or os.path.isfile(full_path):
                        results.append(full_path)
            return results

    @staticmethod
    def find_files(
        directory: str,
        extensions: Optional[List[str]] = None,
        name_contains: Optional[str] = None,
        max_depth: Optional[int] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[str]:
        """
        查找文件

        Args:
            directory: 搜索目录
            extensions: 文件扩展名列表（如 ['.py', '.txt']）
            name_contains: 文件名包含的字符串
            max_depth: 最大搜索深度
            exclude_patterns: 排除的文件模式

        Returns:
            文件路径列表
        """
        directory = os.path.expanduser(directory)
        results = []
        exclude_set = set(exclude_patterns or [])

        def _search(current_dir: str, current_depth: int):
            if max_depth is not None and current_depth > max_depth:
                return

            try:
                for item in os.listdir(current_dir):
                    if item in exclude_set or any(fnmatch(item, p) for p in exclude_set):
                        continue

                    full_path = os.path.join(current_dir, item)

                    if os.path.isfile(full_path):
                        valid = True

                        if extensions:
                            valid = any(full_path.endswith(ext) for ext in extensions)

                        if valid and name_contains:
                            valid = name_contains.lower() in item.lower()

                        if valid:
                            results.append(full_path)

                    elif os.path.isdir(full_path):
                        _search(full_path, current_depth + 1)

            except PermissionError:
                pass

        _search(directory, 0)
        return sorted(results)

    @staticmethod
    def get_file_info(path: str) -> FileInfo:
        """
        获取文件信息

        Args:
            path: 文件路径

        Returns:
            FileInfo 对象
        """
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        stat = os.stat(path)
        name = os.path.basename(path)
        extension = os.path.splitext(name)[1].lower()

        return FileInfo(
            path=path,
            name=name,
            size=stat.st_size,
            created_time=datetime.fromtimestamp(stat.st_ctime),
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            is_directory=os.path.isdir(path),
            extension=extension,
            mime_type=mimetypes.guess_type(path)[0]
        )

    @staticmethod
    def get_dir_size(directory: str) -> int:
        """
        获取目录大小

        Args:
            directory: 目录路径

        Returns:
            目录大小（字节）
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
        return total_size

    @staticmethod
    def build_directory_tree(
        directory: str,
        max_depth: Optional[int] = None,
        include_files: bool = True
    ) -> DirectoryTree:
        """
        构建目录树

        Args:
            directory: 根目录路径
            max_depth: 最大深度
            include_files: 是否包含文件

        Returns:
            DirectoryTree 对象
        """
        directory = os.path.expanduser(directory)
        name = os.path.basename(directory) or directory

        def _build_tree(current_dir: str, current_depth: int) -> DirectoryTree:
            tree = DirectoryTree(
                path=current_dir,
                name=name if current_depth == 0 else os.path.basename(current_dir),
                is_directory=os.path.isdir(current_dir)
            )

            if not tree.is_directory:
                tree.size = os.path.getsize(current_dir)
                return tree

            if max_depth is not None and current_depth >= max_depth:
                return tree

            try:
                items = sorted(os.listdir(current_dir))
                for item in items:
                    item_path = os.path.join(current_dir, item)
                    if os.path.isdir(item_path):
                        child = _build_tree(item_path, current_depth + 1)
                        tree.children.append(child)
                    elif include_files:
                        child = DirectoryTree(
                            path=item_path,
                            name=item,
                            is_directory=False,
                            size=os.path.getsize(item_path)
                        )
                        tree.children.append(child)
            except PermissionError:
                pass

            return tree

        return _build_tree(directory, 0)

    @staticmethod
    def compare_files(file1: str, file2: str) -> bool:
        """
        比较两个文件是否相同

        Args:
            file1: 第一个文件路径
            file2: 第二个文件路径

        Returns:
            文件是否相同
        """
        return filecmp.cmp(file1, file2, shallow=False)

    @staticmethod
    def compute_file_hash(filepath: str, algorithm: str = 'sha256') -> str:
        """
        计算文件哈希值

        Args:
            filepath: 文件路径
            algorithm: 哈希算法（'md5', 'sha1', 'sha256', 'sha512'）

        Returns:
            十六进制哈希值
        """
        hash_func = hashlib.new(algorithm)

        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    @staticmethod
    def compute_directory_hash(directory: str, algorithm: str = 'sha256') -> str:
        """
        计算目录哈希值（基于文件内容）

        Args:
            directory: 目录路径
            algorithm: 哈希算法

        Returns:
            十六进制哈希值
        """
        files = sorted(FileDirOperate.find_files(directory))
        hash_func = hashlib.new(algorithm)

        for filepath in files:
            rel_path = os.path.relpath(filepath, directory)
            hash_func.update(rel_path.encode())
            hash_func.update(FileDirOperate.compute_file_hash(filepath, algorithm).encode())

        return hash_func.hexdigest()

    @staticmethod
    def compress_tar(
        source_dir: str,
        output_path: str,
        compression: str = 'gz',
        exclude_patterns: Optional[List[str]] = None
    ) -> str:
        """
        压缩目录为 tar 文件

        Args:
            source_dir: 源目录
            output_path: 输出文件路径
            compression: 压缩模式（'gz', 'bz2', 'xz', ''）
            exclude_patterns: 排除模式列表

        Returns:
            压缩文件路径
        """
        source_dir = os.path.expanduser(source_dir)
        output_path = os.path.expanduser(output_path)

        FileDirOperate.ensure_parent_dir(output_path)

        mode_map = {
            'gz': 'w:gz',
            'bz2': 'w:bz2',
            'xz': 'w:xz',
            '': 'w'
        }
        mode = mode_map.get(compression, 'w:gz')

        exclude_func = None
        if exclude_patterns:
            def exclude_func(path):
                for pattern in exclude_patterns:
                    if fnmatch(os.path.basename(path), pattern):
                        return True
                return False

        with tarfile.open(output_path, mode) as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir), filter=exclude_func)

        return output_path

    @staticmethod
    def extract_tar(tar_path: str, extract_dir: str) -> str:
        """
        解压 tar 文件

        Args:
            tar_path: tar 文件路径
            extract_dir: 解压目标目录

        Returns:
            解压目录路径
        """
        tar_path = os.path.expanduser(tar_path)
        extract_dir = os.path.expanduser(extract_dir)

        FileDirOperate.ensure_dir(extract_dir)

        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(extract_dir)

        return extract_dir

    @staticmethod
    def compress_zip(
        source_dir: str,
        output_path: str,
        exclude_patterns: Optional[List[str]] = None
    ) -> str:
        """
        压缩目录为 zip 文件

        Args:
            source_dir: 源目录
            output_path: 输出文件路径
            exclude_patterns: 排除模式列表

        Returns:
            压缩文件路径
        """
        source_dir = os.path.expanduser(source_dir)
        output_path = os.path.expanduser(output_path)

        FileDirOperate.ensure_parent_dir(output_path)

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if exclude_patterns and any(fnmatch(file, p) for p in exclude_patterns):
                        continue

                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)

        return output_path

    @staticmethod
    def extract_zip(zip_path: str, extract_dir: str) -> str:
        """
        解压 zip 文件

        Args:
            zip_path: zip 文件路径
            extract_dir: 解压目标目录

        Returns:
            解压目录路径
        """
        zip_path = os.path.expanduser(zip_path)
        extract_dir = os.path.expanduser(extract_dir)

        FileDirOperate.ensure_dir(extract_dir)

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_dir)

        return extract_dir

    @staticmethod
    def get_files_by_size(
        directory: str,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        extensions: Optional[List[str]] = None
    ) -> List[Tuple[str, int]]:
        """
        根据文件大小筛选文件

        Args:
            directory: 目录路径
            min_size: 最小文件大小（字节）
            max_size: 最大文件大小（字节）
            extensions: 文件扩展名过滤

        Returns:
            (文件路径, 文件大小) 列表
        """
        results = []

        for filepath in FileDirOperate.find_files(directory, extensions=extensions):
            size = os.path.getsize(filepath)

            if min_size is not None and size < min_size:
                continue
            if max_size is not None and size > max_size:
                continue

            results.append((filepath, size))

        return sorted(results, key=lambda x: x[1])

    @staticmethod
    def get_files_by_date(
        directory: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        date_type: str = 'modified'
    ) -> List[Tuple[str, datetime]]:
        """
        根据日期筛选文件

        Args:
            directory: 目录路径
            start_date: 开始日期
            end_date: 结束日期
            date_type: 日期类型（'modified', 'created', 'accessed'）

        Returns:
            (文件路径, 日期) 列表
        """
        results = []

        for filepath in FileDirOperate.find_files(directory):
            stat = os.stat(filepath)

            if date_type == 'modified':
                file_date = datetime.fromtimestamp(stat.st_mtime)
            elif date_type == 'created':
                file_date = datetime.fromtimestamp(stat.st_ctime)
            elif date_type == 'accessed':
                file_date = datetime.fromtimestamp(stat.st_atime)
            else:
                file_date = datetime.fromtimestamp(stat.st_mtime)

            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            results.append((filepath, file_date))

        return sorted(results, key=lambda x: x[1])


class FileLock:
    """
    文件锁类

    用于进程间或线程间的文件锁
    """

    def __init__(self, lock_path: str, timeout: Optional[float] = None):
        """
        初始化文件锁

        Args:
            lock_path: 锁文件路径
            timeout: 超时时间（秒）
        """
        self.lock_path = os.path.expanduser(lock_path)
        self.timeout = timeout
        self._lock_file = None

    def acquire(self, blocking: bool = True) -> bool:
        """
        获取锁

        Args:
            blocking: 是否阻塞等待

        Returns:
            是否成功获取锁
        """
        start_time = datetime.now()

        while True:
            try:
                self._lock_file = open(self.lock_path, 'x')
                self._lock_file.write(str(os.getpid()))
                self._lock_file.flush()
                return True
            except FileExistsError:
                if not blocking:
                    return False

                if self.timeout:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed >= self.timeout:
                        return False

        return False

    def release(self) -> None:
        """释放锁"""
        if self._lock_file:
            try:
                self._lock_file.close()
            except Exception:
                pass
            finally:
                self._lock_file = None

        try:
            if os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        except Exception:
            pass

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"获取锁失败: {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


@contextmanager
def file_lock(lock_path: str, timeout: Optional[float] = None):
    """
    文件锁上下文管理器

    Args:
        lock_path: 锁文件路径
        timeout: 超时时间

    Example:
        with file_lock('/tmp/my_app.lock'):
            # 临界区代码
            pass
    """
    lock = FileLock(lock_path, timeout)
    try:
        if not lock.acquire():
            raise TimeoutError(f"获取锁失败: {self.lock_path}")
        yield lock
    finally:
        lock.release()


@contextmanager
def temporary_directory(suffix: str = '', prefix: str = 'tmp', dir: Optional[str] = None):
    """
    创建临时目录上下文管理器

    Args:
        suffix: 目录名后缀
        prefix: 目录名前缀
        dir: 临时目录父目录

    Example:
        with temporary_directory() as tmpdir:
            # 使用临时目录
            pass
    """
    tmpdir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextmanager
def temporary_file(suffix: str = '', prefix: str = 'tmp', dir: Optional[str] = None):
    """
    创建临时文件上下文管理器

    Args:
        suffix: 文件名后缀
        prefix: 文件名前缀
        dir: 临时文件目录

    Example:
        with temporary_file(suffix='.txt') as tmpfile:
            # 使用临时文件
            pass
    """
    fd, tmpfile = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    try:
        os.close(fd)
        yield tmpfile
    finally:
        try:
            os.remove(tmpfile)
        except FileNotFoundError:
            pass


def ensure_dir(path: str) -> str:
    """
    便捷函数：确保目录存在

    Args:
        path: 目录路径

    Returns:
        目录路径
    """
    return FileDirOperate.ensure_dir(path)


def list_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = False
) -> List[str]:
    """
    便捷函数：列出文件

    Args:
        directory: 目录路径
        pattern: 文件模式
        recursive: 是否递归

    Returns:
        文件路径列表
    """
    return FileDirOperate.list_files(directory, pattern, recursive)


def copy_file(src: str, dst: str, overwrite: bool = False) -> str:
    """
    便捷函数：复制文件

    Args:
        src: 源文件
        dst: 目标文件
        overwrite: 是否覆盖

    Returns:
        目标文件路径
    """
    return FileDirOperate.copy_file(src, dst, overwrite)


def move_file(src: str, dst: str, overwrite: bool = False) -> str:
    """
    便捷函数：移动文件

    Args:
        src: 源文件
        dst: 目标文件
        overwrite: 是否覆盖

    Returns:
        目标文件路径
    """
    return FileDirOperate.move_file(src, dst, overwrite)


def delete_file(path: str, missing_ok: bool = False) -> bool:
    """
    便捷函数：删除文件

    Args:
        path: 文件路径
        missing_ok: 缺失是否忽略

    Returns:
        是否成功
    """
    return FileDirOperate.delete_file(path, missing_ok)


def get_file_size(path: str) -> int:
    """
    便捷函数：获取文件大小

    Args:
        path: 文件路径

    Returns:
        文件大小（字节）
    """
    return os.path.getsize(os.path.expanduser(path))


def format_size(size_bytes: int) -> str:
    """
    格式化文件大小为人类可读格式

    Args:
        size_bytes: 字节数

    Returns:
        格式化的大小字符串

    Example:
        >>> format_size(1024)
        '1.00 KB'
        >>> format_size(1024 * 1024)
        '1.00 MB'
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


def get_unique_filename(filepath: str) -> str:
    """
    获取不重复的文件名（添加数字后缀）

    Args:
        filepath: 文件路径

    Returns:
        不重复的文件路径
    """
    filepath = os.path.expanduser(filepath)

    if not os.path.exists(filepath):
        return filepath

    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)

    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_path = os.path.join(directory, new_filename) if directory else new_filename
        if not os.path.exists(new_path):
            return new_path
        counter += 1
