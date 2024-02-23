from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
from json import dumps

def colorize_json(data):
    json_str = dumps(data, indent=4)
    colorful_json = highlight(json_str, JsonLexer(), TerminalFormatter())
    return colorful_json