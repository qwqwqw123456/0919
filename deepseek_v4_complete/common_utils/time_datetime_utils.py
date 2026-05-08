"""
时间日期工具模块

提供完整的时间和日期处理功能，包括：
- 日期时间解析和格式化
- 时区转换
- 时间间隔计算
- 定时任务调度
- 时间戳处理
- 友好时间显示
"""

import os
import time
import calendar
from datetime import datetime, timedelta, timezone, date
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class TimeFormat(Enum):
    """常用时间格式枚举"""
    ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"
    ISO_FORMAT_MS = "%Y-%m-%dT%H:%M:%S.%f"
    DATE_ONLY = "%Y-%m-%d"
    TIME_ONLY = "%H:%M:%S"
    DATETIME_DISPLAY = "%Y-%m-%d %H:%M:%S"
    DATETIME_CN = "%Y年%m月%d日 %H:%M:%S"
    DATE_CN = "%Y年%m月%d日"
    TIMESTAMP = "%Y%m%d_%H%M%S"
    TIMESTAMP_MS = "%Y%m%d_%H%M%S_%f"


@dataclass
class TimeRange:
    """时间范围数据类"""
    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        """时间范围持续时长"""
        return self.end - self.start

    @property
    def duration_seconds(self) -> float:
        """时间范围持续秒数"""
        return self.duration.total_seconds()

    def contains(self, dt: datetime) -> bool:
        """检查时间是否在范围内"""
        return self.start <= dt <= self.end

    def overlaps(self, other: 'TimeRange') -> bool:
        """检查两个时间范围是否重叠"""
        return self.start <= other.end and self.end >= other.start


class TimeDateUtils:
    """
    时间日期工具类

    提供静态方法和便捷函数
    """

    @staticmethod
    def now(timezone_id: Optional[str] = None) -> datetime:
        """
        获取当前时间

        Args:
            timezone_id: 时区 ID（如 'Asia/Shanghai', 'UTC'）

        Returns:
            当前 datetime 对象
        """
        if timezone_id:
            tz = TimeDateUtils.get_timezone(timezone_id)
            return datetime.now(tz)
        return datetime.now()

    @staticmethod
    def utc_now() -> datetime:
        """获取当前 UTC 时间"""
        return datetime.now(timezone.utc)

    @staticmethod
    def today(timezone_id: Optional[str] = None) -> date:
        """
        获取今天的日期

        Args:
            timezone_id: 时区 ID

        Returns:
            date 对象
        """
        return TimeDateUtils.now(timezone_id).date()

    @staticmethod
    def get_timestamp(dt: Optional[datetime] = None, milliseconds: bool = True) -> int:
        """
        获取时间戳

        Args:
            dt: datetime 对象，None 表示当前时间
            milliseconds: 是否返回毫秒级时间戳

        Returns:
            时间戳
        """
        if dt is None:
            dt = datetime.now()

        if milliseconds:
            return int(dt.timestamp() * 1000)
        return int(dt.timestamp())

    @staticmethod
    def from_timestamp(timestamp: int, timezone_id: Optional[str] = None) -> datetime:
        """
        从时间戳创建 datetime

        Args:
            timestamp: 时间戳（毫秒或秒）
            timezone_id: 时区 ID

        Returns:
            datetime 对象
        """
        if timestamp > 10**12:
            timestamp = timestamp / 1000

        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        if timezone_id:
            target_tz = TimeDateUtils.get_timezone(timezone_id)
            dt = dt.astimezone(target_tz)

        return dt

    @staticmethod
    def parse(
        date_string: str,
        format_string: Optional[str] = None,
        timezone_id: Optional[str] = None
    ) -> datetime:
        """
        解析日期字符串

        Args:
            date_string: 日期字符串
            format_string: 格式字符串
            timezone_id: 时区 ID

        Returns:
            datetime 对象

        Raises:
            ValueError: 解析失败
        """
        if format_string:
            dt = datetime.strptime(date_string, format_string)
        else:
            dt = TimeDateUtils._auto_parse(date_string)

        if timezone_id:
            tz = TimeDateUtils.get_timezone(timezone_id)
            dt = dt.replace(tzinfo=tz)

        return dt

    @staticmethod
    def _auto_parse(date_string: str) -> datetime:
        """自动解析日期字符串"""
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y%m%d%H%M%S",
            "%Y%m%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        try:
            from dateutil import parser
            return parser.parse(date_string)
        except (ImportError, Exception):
            pass

        raise ValueError(f"无法解析日期字符串: {date_string}")

    @staticmethod
    def format(
        dt: Optional[datetime] = None,
        format_string: str = "%Y-%m-%d %H:%M:%S"
    ) -> str:
        """
        格式化时间

        Args:
            dt: datetime 对象，None 表示当前时间
            format_string: 格式字符串

        Returns:
            格式化的时间字符串
        """
        if dt is None:
            dt = datetime.now()
        return dt.strftime(format_string)

    @staticmethod
    def format_iso(dt: Optional[datetime] = None, include_microseconds: bool = False) -> str:
        """
        格式化为 ISO 格式

        Args:
            dt: datetime 对象
            include_microseconds: 是否包含微秒

        Returns:
            ISO 格式字符串
        """
        if dt is None:
            dt = datetime.now()
        if include_microseconds:
            return dt.isoformat()
        return dt.replace(microsecond=0).isoformat()

    @staticmethod
    def format_relative(dt: Optional[datetime] = None, reference: Optional[datetime] = None) -> str:
        """
        格式化为相对时间（友好时间）

        Args:
            dt: 要格式化的时间，None 表示当前时间
            reference: 参考时间，None 表示当前时间

        Returns:
            相对时间字符串

        Example:
            >>> format_relative(datetime.now() - timedelta(minutes=5))
            '5分钟前'
        """
        if dt is None:
            dt = datetime.now()
        if reference is None:
            reference = datetime.now()

        diff = reference - dt

        seconds = diff.total_seconds()

        if seconds < 0:
            diff = -diff
            seconds = diff.total_seconds()
            suffix = "后"
        else:
            suffix = "前"

        if seconds < 60:
            return f"{int(seconds)}秒{suffix}"
        elif seconds < 3600:
            return f"{int(seconds / 60)}分钟{suffix}"
        elif seconds < 86400:
            return f"{int(seconds / 3600)}小时{suffix}"
        elif seconds < 2592000:
            return f"{int(seconds / 86400)}天{suffix}"
        elif seconds < 31536000:
            return f"{int(seconds / 2592000)}个月{suffix}"
        else:
            return f"{int(seconds / 31536000)}年{suffix}"

    @staticmethod
    def get_timezone(timezone_id: str) -> timezone:
        """
        获取时区对象

        Args:
            timezone_id: 时区 ID（如 'Asia/Shanghai', 'UTC'）

        Returns:
            timezone 对象
        """
        import pytz
        return pytz.timezone(timezone_id)

    @staticmethod
    def convert_timezone(
        dt: datetime,
        from_tz: Optional[str] = None,
        to_tz: str = "Asia/Shanghai"
    ) -> datetime:
        """
        转换时区

        Args:
            dt: datetime 对象
            from_tz: 源时区 ID
            to_tz: 目标时区 ID

        Returns:
            转换后的 datetime 对象
        """
        if from_tz:
            from_timezone = TimeDateUtils.get_timezone(from_tz)
            if dt.tzinfo is None:
                dt = from_timezone.localize(dt)
            else:
                dt = dt.astimezone(from_timezone)

        target_tz = TimeDateUtils.get_timezone(to_tz)
        return dt.astimezone(target_tz)

    @staticmethod
    def add_time(
        dt: Optional[datetime] = None,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0
    ) -> datetime:
        """
        添加时间

        Args:
            dt: datetime 对象，None 表示当前时间
            days: 天数
            hours: 小时数
            minutes: 分钟数
            seconds: 秒数
            microseconds: 微秒数

        Returns:
            新的 datetime 对象
        """
        if dt is None:
            dt = datetime.now()

        delta = timedelta(
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds
        )
        return dt + delta

    @staticmethod
    def subtract_time(
        dt: Optional[datetime] = None,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0
    ) -> datetime:
        """
        减去时间

        Args:
            dt: datetime 对象，None 表示当前时间
            其他参数同上

        Returns:
            新的 datetime 对象
        """
        return TimeDateUtils.add_time(
            dt, days=-days, hours=-hours, minutes=-minutes,
            seconds=-seconds, microseconds=-microseconds
        )

    @staticmethod
    def get_time_range(
        period: str,
        reference: Optional[datetime] = None
    ) -> TimeRange:
        """
        获取时间范围

        Args:
            period: 时间段类型（'today', 'yesterday', 'this_week', 'last_week', 'this_month', 'last_month', 'this_year'）
            reference: 参考时间

        Returns:
            TimeRange 对象
        """
        if reference is None:
            reference = datetime.now()

        if period == 'today':
            start = reference.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(microseconds=1)

        elif period == 'yesterday':
            start = (reference - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(microseconds=1)

        elif period == 'this_week':
            start = reference - timedelta(days=reference.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7) - timedelta(microseconds=1)

        elif period == 'last_week':
            end = reference - timedelta(days=reference.weekday())
            end = end.replace(hour=0, minute=0, second=0, microsecond=0)
            start = end - timedelta(days=7)
            end = end - timedelta(microseconds=1)

        elif period == 'this_month':
            start = reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = start.replace(month=start.month + 1) if start.month < 12 else start.replace(year=start.year + 1, month=1)
            end = next_month - timedelta(microseconds=1)

        elif period == 'last_month':
            first_this_month = reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = first_this_month - timedelta(microseconds=1)
            start = end.replace(day=1)

        elif period == 'this_year':
            start = reference.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(year=start.year + 1) - timedelta(microseconds=1)

        else:
            start = reference.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(microseconds=1)

        return TimeRange(start=start, end=end)

    @staticmethod
    def get_week_range(reference: Optional[datetime] = None, week_start: int = 0) -> TimeRange:
        """
        获取指定周的日期范围

        Args:
            reference: 参考时间
            week_start: 周起始日（0=周一，1=周二...）

        Returns:
            TimeRange 对象
        """
        if reference is None:
            reference = datetime.now()

        days_to_subtract = (reference.weekday() - week_start) % 7
        start = reference - timedelta(days=days_to_subtract)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7) - timedelta(microseconds=1)

        return TimeRange(start=start, end=end)

    @staticmethod
    def get_month_range(
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> TimeRange:
        """
        获取指定月份的日期范围

        Args:
            year: 年份，None 表示当前年份
            month: 月份，None 表示当前月份

        Returns:
            TimeRange 对象
        """
        now = datetime.now()

        if year is None:
            year = now.year
        if month is None:
            month = now.month

        start = datetime(year, month, 1)

        if month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(microseconds=1)
        else:
            end = datetime(year, month + 1, 1) - timedelta(microseconds=1)

        return TimeRange(start=start, end=end)

    @staticmethod
    def is_business_day(dt: Optional[datetime] = None) -> bool:
        """
        判断是否为工作日

        Args:
            dt: datetime 对象，None 表示今天

        Returns:
            是否为工作日
        """
        if dt is None:
            dt = datetime.now()

        return dt.weekday() < 5

    @staticmethod
    def is_weekend(dt: Optional[datetime] = None) -> bool:
        """
        判断是否为周末

        Args:
            dt: datetime 对象，None 表示今天

        Returns:
            是否为周末
        """
        if dt is None:
            dt = datetime.now()

        return dt.weekday() >= 5

    @staticmethod
    def get_business_days_count(
        start: datetime,
        end: datetime,
        exclude_holidays: Optional[List[date]] = None
    ) -> int:
        """
        计算工作日数量

        Args:
            start: 开始日期
            end: 结束日期
            exclude_holidays: 需要排除的假期日期列表

        Returns:
            工作日数量
        """
        days = 0
        current = start
        holidays = set(exclude_holidays or [])

        while current <= end:
            if current.weekday() < 5 and current.date() not in holidays:
                days += 1
            current += timedelta(days=1)

        return days

    @staticmethod
    def get_next_business_day(dt: Optional[datetime] = None) -> datetime:
        """
        获取下一个工作日

        Args:
            dt: datetime 对象，None 表示今天

        Returns:
            下一个工作日的 datetime
        """
        if dt is None:
            dt = datetime.now()

        next_day = dt + timedelta(days=1)

        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        return next_day

    @staticmethod
    def get_age(birth_date: date, reference: Optional[date] = None) -> int:
        """
        计算年龄

        Args:
            birth_date: 出生日期
            reference: 参考日期，None 表示今天

        Returns:
            年龄
        """
        if reference is None:
            reference = date.today()

        age = reference.year - birth_date.year

        if (reference.month, reference.day) < (birth_date.month, birth_date.day):
            age -= 1

        return age

    @staticmethod
    def get_quarter(dt: Optional[datetime] = None) -> int:
        """
        获取季度

        Args:
            dt: datetime 对象，None 表示当前时间

        Returns:
            季度（1-4）
        """
        if dt is None:
            dt = datetime.now()
        return (dt.month - 1) // 3 + 1

    @staticmethod
    def get_quarter_range(
        year: Optional[int] = None,
        quarter: Optional[int] = None
    ) -> TimeRange:
        """
        获取指定季度的日期范围

        Args:
            year: 年份，None 表示当前年份
            quarter: 季度（1-4），None 表示当前季度

        Returns:
            TimeRange 对象
        """
        now = datetime.now()

        if year is None:
            year = now.year
        if quarter is None:
            quarter = TimeDateUtils.get_quarter(now)

        start_month = (quarter - 1) * 3 + 1
        start = datetime(year, start_month, 1)

        if quarter == 4:
            end = datetime(year + 1, 1, 1) - timedelta(microseconds=1)
        else:
            end = datetime(year, start_month + 3, 1) - timedelta(microseconds=1)

        return TimeRange(start=start, end=end)

    @staticmethod
    def format_duration(seconds: float, precision: int = 2) -> str:
        """
        格式化时长

        Args:
            seconds: 秒数
            precision: 小数精度

        Returns:
            格式化的时间字符串

        Example:
            >>> format_duration(3665)
            '1小时1分钟5秒'
        """
        if seconds < 0:
            return "0秒"

        parts = []
        units = [
            (86400, "天"),
            (3600, "小时"),
            (60, "分钟"),
            (1, "秒")
        ]

        for unit_seconds, unit_name in units:
            if seconds >= unit_seconds:
                value = int(seconds // unit_seconds)
                seconds -= value * unit_seconds
                if precision > 0 and unit_name == "秒" and seconds > 0:
                    parts.append(f"{value}.{int(seconds * (10 ** precision))}{unit_name}")
                else:
                    parts.append(f"{value}{unit_name}")

        if not parts:
            return f"0.{int(seconds * (10 ** precision))}秒"

        return "".join(parts)

    @staticmethod
    def parse_duration(duration_str: str) -> float:
        """
        解析时长字符串

        Args:
            duration_str: 时长字符串

        Returns:
            秒数

        Example:
            >>> parse_duration("1h30m")
            5400.0
            >>> parse_duration("2 days 3 hours")
            176400.0
        """
        seconds = 0.0
        current_num = ""

        duration_str = duration_str.lower().strip()

        for char in duration_str:
            if char.isdigit() or char == '.':
                current_num += char
            elif char in 'dhms':
                if current_num:
                    num = float(current_num)
                    if char == 'd':
                        seconds += num * 86400
                    elif char == 'h':
                        seconds += num * 3600
                    elif char == 'm':
                        seconds += num * 60
                    elif char == 's':
                        seconds += num
                    current_num = ""
            elif char == ' ':
                if current_num:
                    try:
                        seconds += float(current_num)
                    except ValueError:
                        pass
                    current_num = ""

        if current_num:
            try:
                seconds += float(current_num)
            except ValueError:
                pass

        return seconds

    @staticmethod
    def get_weekday_name(weekday: int, short: bool = False) -> str:
        """
        获取星期几的名称

        Args:
            weekday: 星期几（0=周一，6=周日）
            short: 是否返回短名称

        Returns:
            星期名称
        """
        if short:
            names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        else:
            names = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
        return names[weekday]

    @staticmethod
    def get_month_name(month: int, short: bool = False) -> str:
        """
        获取月份的名称

        Args:
            month: 月份（1-12）
            short: 是否返回短名称

        Returns:
            月份名称
        """
        if short:
            names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
        else:
            names = ['一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月']
        return names[month - 1]

    @staticmethod
    def sleep(seconds: float) -> None:
        """
        睡眠指定秒数

        Args:
            seconds: 秒数（支持小数）
        """
        time.sleep(seconds)

    @staticmethod
    def wait_until(target_time: Union[datetime, str], check_interval: float = 1.0) -> None:
        """
        等待直到指定时间

        Args:
            target_time: 目标时间（datetime 对象或时间字符串）
            check_interval: 检查间隔（秒）
        """
        if isinstance(target_time, str):
            target_time = TimeDateUtils.parse(target_time)

        while datetime.now() < target_time:
            time.sleep(check_interval)


def now_str(format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    便捷函数：获取当前时间的格式化字符串

    Args:
        format_string: 格式字符串

    Returns:
        格式化的时间字符串
    """
    return datetime.now().strftime(format_string)


def today_str(format_string: str = "%Y-%m-%d") -> str:
    """
    便捷函数：获取今天日期的格式化字符串

    Args:
        format_string: 格式字符串

    Returns:
        格式化的日期字符串
    """
    return date.today().strftime(format_string)


def timestamp() -> int:
    """
    便捷函数：获取当前时间戳（毫秒）

    Returns:
        时间戳
    """
    return int(time.time() * 1000)


def timestamp_seconds() -> int:
    """
    便捷函数：获取当前时间戳（秒）

    Returns:
        时间戳
    """
    return int(time.time())


def get_date_range(
    start_date: Union[str, date],
    end_date: Union[str, date]
) -> List[date]:
    """
    便捷函数：获取日期范围内的所有日期

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        日期列表
    """
    if isinstance(start_date, str):
        start_date = TimeDateUtils.parse(start_date).date()
    if isinstance(end_date, str):
        end_date = TimeDateUtils.parse(end_date).date()

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    return dates


def format_time(seconds: float) -> str:
    """
    便捷函数：格式化时长

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    return TimeDateUtils.format_duration(seconds)


def parse_time(duration_str: str) -> float:
    """
    便捷函数：解析时长字符串

    Args:
        duration_str: 时长字符串

    Returns:
        秒数
    """
    return TimeDateUtils.parse_duration(duration_str)


def is_same_day(dt1: datetime, dt2: datetime) -> bool:
    """
    便捷函数：判断两个时间是否是同一天

    Args:
        dt1: 第一个 datetime
        dt2: 第二个 datetime

    Returns:
        是否是同一天
    """
    return dt1.year == dt2.year and dt1.month == dt2.month and dt1.day == dt2.day


def get_days_between(start_date: date, end_date: date) -> int:
    """
    便捷函数：获取两个日期之间的天数

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        天数
    """
    return (end_date - start_date).days


class Timer:
    """
    计时器类

    用于测量代码执行时间
    """

    def __init__(self, name: Optional[str] = None, verbose: bool = True):
        """
        初始化计时器

        Args:
            name: 计时器名称
            verbose: 是否自动打印结果
        """
        self.name = name or "Timer"
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def start(self) -> 'Timer':
        """开始计时"""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed = None
        return self

    def stop(self) -> float:
        """
        停止计时

        Returns:
            经过的秒数
        """
        if self.start_time is None:
            raise RuntimeError("计时器未启动")

        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

        if self.verbose:
            print(f"{self.name}: {TimeDateUtils.format_duration(self.elapsed)}")

        return self.elapsed

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        return False

    def reset(self) -> None:
        """重置计时器"""
        self.start_time = None
        self.end_time = None
        self.elapsed = None
