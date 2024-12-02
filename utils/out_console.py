import time
from enum import IntEnum, unique


@unique
class Style(IntEnum):
    DEFAULT = 0
    BOLD = 1
    ITALIC = 3
    UNDERLINE = 4
    BLINK = 5
    ANTIWHITE = 7


@unique
class Color(IntEnum):
    DEFAULT = 39
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    PURPLE = 35
    CYAN = 36
    WHITE = 37
    LIGHTBLACK_EX = 90
    LIGHTRED_EX = 91
    LIGHTGREEN_EX = 92
    LIGHTYELLOW_EX = 93
    LIGHTBLUE_EX = 94
    LIGHTMAGENTA_EX = 95
    LIGHTCYAN_EX = 96
    LIGHTWHITE_EX = 97


@unique
class BGColor(IntEnum):
    DEFAULT = 49
    BLACK = 40
    RED = 41
    GREEN = 42
    YELLOW = 43
    BLUE = 44
    PURPLE = 45
    CYAN = 46
    WHITE = 47
    LIGHTBLACK_EX = 100
    LIGHTRED_EX = 101
    LIGHTGREEN_EX = 102
    LIGHTYELLOW_EX = 103
    LIGHTBLUE_EX = 104
    LIGHTMAGENTA_EX = 105
    LIGHTCYAN_EX = 106
    LIGHTWHITE_EX = 107


class OutConsole:
    def __init__(self):
        pass

    @staticmethod
    def out(
            content: str,
            color: int = Color.DEFAULT.value,
            bg_color: int = BGColor.DEFAULT.value,
            style: int = Style.DEFAULT.value
    ) -> None:
        print("\033[{};{};{}m{}\033[0m".format(style, color, bg_color, content))

    @staticmethod
    def format_time(value: float) -> str:
        int_x = int(value)
        h = int(int_x / 3600)
        m = (int_x % 3600) // 60
        s = int_x - h * 3600 - m * 60
        label_time = format(h, '02d') + ':' + format(m, '02d') + ':' + format(s, '02d')
        return label_time

    def out_title(
            self,
            title: str,
            color: int = Color.RED.value,
            style: int = Style.BOLD.value,
            level: int = 0
    ) -> None:
        title = ' ' * 4 * level + title
        self.out(title, color=color, style=style)

    def out_line(
            self,
            title: str,
            mark: str = '  ',
            color: int = Color.RED.value,
            style: int = Style.BOLD.value,
            length: int = 160
    ) -> None:
        label = ' ' * 2 + title + ' ' * 2
        if len(label) % 2 != 0:
            label += ' '
        half = int((length - len(label)) / 2)
        line = format('', f'{mark[0]}<{half}') + label + format('', f'{mark[1]}>{half}')
        self.out(content=line, color=color, style=style)

    def repeat(
            self,
            mark: str = '  ',
            length: int = 160,
            color: int = Color.GREEN.value
    ):
        def decorator_repeat(func):  # 这个是之前的外层函数，接收func，返回经装饰后的func.
            def wrapper_repeat():
                start = time.time()
                time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                self.out_line(time_str, mark=mark, length=length, color=color, style=Style.BOLD.value)
                ret = func()
                time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                self.out_line(time_str, mark=mark, length=length, color=color, style=Style.BOLD.value)
                time_str = self.format_time(time.time() - start)
                self.out_line(time_str, mark='  ', length=length, color=color, style=Style.BOLD.value)
                return ret

            return wrapper_repeat

        return decorator_repeat


out_console = OutConsole()

if __name__ == '__main__':
    out_console.out_line('black', mark='<>', color=Color.RED.value, style=Style.BOLD.value, length=120)
