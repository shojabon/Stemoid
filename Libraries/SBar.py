class SBar:

    def __init__(self, max_num=30, empty_char='-', used_char='#'):
        self._max_num = int(max_num)
        self._empty_char = empty_char
        self._used_char = used_char

    def render_bar(self, percentage, show_num_char_num=6, label=None):
        if percentage > self._max_num: percentage = self._max_num
        if label is None: label=''
        string = str(label) + self.padding_text(str(percentage*100), show_num_char_num) + str("%") + '[' + '#' * int(percentage*self._max_num) + '-' * (self._max_num - int(percentage*self._max_num)) + ']'
        return string

    def padding_text(self, text, characters):
        num = len(str(text))
        if characters <= num: num = characters
        return str(text)[:characters] + ' ' * (characters-num)