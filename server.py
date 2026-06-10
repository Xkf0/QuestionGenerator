"""
AI 口算练习 - 本地服务器
启动后访问 http://localhost:3000
"""
import http.server
import json
import os
import sys
import re
import urllib.request
import urllib.error
from fractions import Fraction

PORT = 3000
STATS_FILE = os.path.join(os.path.dirname(__file__), 'stats-data.json')


def load_stats() -> dict:
    """读取统计数据"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, IOError):
            pass
    return {'dailyStats': {}}


def save_stats(data: dict):
    """保存统计数据"""
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f'[Stats] 写入失败: {e}', file=sys.stderr)


def call_deepseek(api_key: str, prompt: str) -> str | None:
    """调用 DeepSeek API 并返回文本内容"""
    data = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000,
    }).encode('utf-8')

    req = urllib.request.Request(
        'https://api.deepseek.com/v1/chat/completions',
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode('utf-8'))
        content = body.get('choices', [{}])[0].get('message', {}).get('content', '')
        return content.strip() if content else None
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        print(f'[API Error] {e}', file=sys.stderr)
        return None


def parse_json_from_response(text: str) -> dict | None:
    """从 AI 返回文本中提取 JSON 对象"""
    # 尝试直接解析
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # 尝试从 ```json ... ``` 中提取
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    return None


def max_digits_of_parts(n: str) -> int:
    """返回数字各部分（整数、小数、分子、分母）的最大位数"""
    clean = n.lstrip('-').lstrip('+')
    if '/' in clean:
        # 分数：检查分子和分母各自位数
        parts = clean.split('/')
        return max(len(p.replace('.', '')) for p in parts)
    if '.' in clean:
        # 小数：分别检查整数部分和小数部分
        int_part, dec_part = clean.split('.')
        return max(len(int_part), len(dec_part))
    return len(clean)


def validate_numbers_in_expr(expr: str) -> bool:
    """验证表达式中的所有数字，每部分最多2位"""
    tokens = re.findall(r'-?\d+(?:\.\d+)?(?:/-?\d+(?:\.\d+)?)?', expr)
    for t in tokens:
        if max_digits_of_parts(t) > 2:
            return False
    return True


def check_type_consistency(expr: str) -> str | None:
    """检查是否混用小数和分数，返回表达式类型（integer/decimal/fraction）或 None（混用）"""
    has_fraction = bool(re.search(r'(?<!\w)\d+/\d+', expr))
    has_decimal = bool(re.search(r'\d+\.\d+', expr))

    if has_fraction and has_decimal:
        return None  # 混用，不合法
    if has_fraction:
        return 'fraction'
    if has_decimal:
        return 'decimal'
    return 'integer'


def count_operators(expr: str) -> bool:
    """验证运算符数量不超过限制"""
    adds = expr.count('+') - expr.count('1e')  # 排除科学计数法（不会出现）
    subs = expr.count('-')
    mults = expr.count('*')
    divs = expr.count('/')
    # 修正：负数符号不算减法
    # 去掉分数中的 / 号
    # 简单的启发式：去掉所有数字和括号后的 +-*/ 才算运算符
    cleaned = re.sub(r'-?\d+(?:\.\d+)?(?:/-?\d+(?:\.\d+)?)?', '', expr)
    cleaned = re.sub(r'[()\s]', '', cleaned)

    adds = cleaned.count('+')
    subs = cleaned.count('-')
    mults = cleaned.count('*')
    divs = cleaned.count('/')

    return adds <= 2 and subs <= 2 and mults <= 2 and divs <= 2


def safe_eval(expr: str) -> Fraction | None:
    """安全计算表达式，返回精确的 Fraction 结果"""
    expr = expr.replace('×', '*').replace('÷', '/')
    # 只允许数字、运算符、括号、小数点
    if not re.match(r'^[\d+\-*/().\s]+$', expr):
        return None
    try:
        # 将 a/b 形式的分数替换为 Fraction(a,b)，确保精确计算
        # 注意：替换后表达式中的 / 将只用于 Fraction 构造，不再作为除法运算符
        eval_expr = re.sub(r'(-?\d+\.?\d*)/(-?\d+\.?\d*)', r'Fraction(\1,\2)', expr)
        # 还支持正号前缀
        eval_expr = re.sub(r'(\+?\d+\.?\d*)/(\+?\d+\.?\d*)', r'Fraction(\1,\2)', eval_expr)
        # 处理多余的替换（如 -(Fraction...) 的情况）
        eval_expr = re.sub(r'\(Fraction\(', '(Fraction(', eval_expr)

        result = eval(eval_expr, {'__builtins__': {}}, {'Fraction': Fraction})
        if isinstance(result, (int, float)):
            return Fraction(result).limit_denominator(1000000)
        if isinstance(result, Fraction):
            return result
        return None
    except (ZeroDivisionError, ArithmeticError, SyntaxError, TypeError, ValueError) as e:
        print(f'[safe_eval] Error: {e}', file=sys.stderr)
        return None


def validate_result(result: Fraction) -> (bool, str):
    """验证计算结果是否符合约束，返回 (是否合法, 格式化后的答案)"""
    # 检查范围
    if result < -1000 or result > 1000:
        return False, ''

    # 如果是整数
    if result.denominator == 1:
        val = int(result)
        if abs(val) >= 1000:
            return False, ''
        return True, str(val)

    # 检查分数分子分母位数
    num_digits = len(str(abs(result.numerator)))
    den_digits = len(str(result.denominator))
    if num_digits > 3 or den_digits > 3:
        return False, ''

    # 检查是否可以表示为有限小数且小数位 ≤ 3
    dec = float(result)
    if abs(dec) < 1000:
        # 检查小数位
        dec_str = f'{dec:.10f}'.rstrip('0')
        if '.' in dec_str and len(dec_str.split('.')[1]) <= 3:
            return True, str(dec)
        # 有限小数但位数过多，看简化后分母是否能被 2/5 整除
        # 简化分数再检查
        return True, f'{result.numerator}/{result.denominator}'

    return True, f'{result.numerator}/{result.denominator}'


SYSTEM_PROMPT = """你是一个口算题生成器。请生成一道符合以下全部约束的口算题：

数字约束：
- 题目中出现的每个数字最多2位（包括整数部分和小数部分）
- 例如允许：5, 63, 13.2, 23, 23.3, 32/3, 32/63
- 例如不允许：123, 1.234, 123/456

运算约束：
- 最多2个加法(+)、2个减法(-)、2个乘法(×)、2个除法(÷)
- 可以使用括号改变运算顺序
- 运算符总数不超过8个

结果约束：
- 计算结果必须在 -1000 到 1000 之间
- 如果是小数，小数点后最多3位
- 如果是分数，分子和分母各自最多3位数字

类型一致性（重要！）：
- 题目中不能同时出现小数和分数，只能选择一种
- 如果题目中有分数（如 2/3），答案必须用最简分数表示（如 1/2 而非 2/4）
- 如果题目中有小数（如 1.5），答案必须用小数表示
- 如果题目中只有整数，答案用整数表示

请严格按以下 JSON 格式返回，不要包含其他内容：
{
  "question": "题目文本（用 × 和 ÷ 符号）",
  "expression": "可用于 Python eval 的表达式（用 * 和 /）",
  "answer": "标准答案（如 68 或 697/63 或 23.456）",
  "answerType": "integer | decimal | fraction",
  "solution": "详细的解题步骤，用中文，分步说明，每步单独一行"
}"""


def generate_question(api_key: str) -> dict | None:
    """生成一道符合约束的口算题，最多重试3次"""
    for attempt in range(3):
        print(f'[Question] 尝试生成第 {attempt + 1} 次...')
        raw = call_deepseek(api_key, SYSTEM_PROMPT)
        if not raw:
            continue

        data = parse_json_from_response(raw)
        if not data:
            continue

        question = data.get('question', '').strip()
        expression = data.get('expression', '').strip()
        answer = data.get('answer', '').strip()
        answer_type = data.get('answerType', '').strip()
        solution = data.get('solution', '').strip()

        if not all([question, expression, answer, answer_type, solution]):
            print(f'[Question] 字段不完整，跳过')
            continue

        # 验证数字位数
        if not validate_numbers_in_expr(expression):
            print(f'[Question] 数字位数超标，跳过')
            continue

        # 验证运算符数量
        if not count_operators(expression):
            print(f'[Question] 运算符数量超标，跳过')
            continue

        # 验证类型一致性：不能同时出现小数和分数
        expr_type = check_type_consistency(expression)
        if expr_type is None:
            print(f'[Question] 混用了小数和分数，跳过')
            continue

        # 验证 answerType 与表达式类型一致
        if answer_type != expr_type and expr_type != 'integer':
            print(f'[Question] answerType ({answer_type}) 与表达式类型 ({expr_type}) 不一致，跳过')
            continue

        # 计算结果并验证
        result = safe_eval(expression)
        if result is None:
            print(f'[Question] 表达式计算失败，跳过')
            continue

        valid, formatted = validate_result(result)
        if not valid:
            print(f'[Question] 结果不符合约束，跳过')
            continue

        # 使用服务器计算的最简答案覆盖 AI 给的答案
        server_answer = formatted
        if expr_type == 'fraction':
            # 确保分数是最简形式
            server_answer = f'{result.numerator}/{result.denominator}'
        elif expr_type == 'decimal' and result.denominator == 1:
            # 纯整数但表达式有小数 → 结果也是整数
            server_answer = str(int(result))

        print(f'[Question] 生成成功: {question}')
        return {
            'question': question,
            'expression': expression,
            'answer': server_answer,
            'answerType': answer_type,
            'solution': solution,
        }

    return None


def verify_answer(expression: str, user_answer: str) -> dict:
    """核验用户答案是否正确"""
    result = safe_eval(expression)
    if result is None:
        return {'correct': False, 'error': '无法计算表达式'}

    # 精确答案
    exact = f'{result.numerator}/{result.denominator}' if result.denominator != 1 else str(int(result))
    exact_float = float(result)

    # 解析用户答案
    user_ans = user_answer.strip()
    try:
        if '/' in user_ans:
            # 分数格式
            parts = user_ans.split('/')
            if len(parts) == 2:
                user_val = Fraction(parts[0].strip()) / Fraction(parts[1].strip())
            else:
                user_val = Fraction(user_ans)
        elif '.' in user_ans:
            user_val = Fraction(user_ans)
        else:
            user_val = Fraction(int(user_ans), 1)

        # 比较
        correct = (user_val == result)
        if not correct and result.denominator != 1:
            # 也允许小数近似比较
            correct = abs(float(user_val) - exact_float) < 0.001

        return {
            'correct': correct,
            'exactAnswer': exact,
            'exactFloat': round(exact_float, 10),
        }
    except (ValueError, ZeroDivisionError):
        return {'correct': False, 'error': '无法解析你的答案'}


class Handler(http.server.SimpleHTTPRequestHandler):

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _read_body(self) -> dict:
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            return json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            return {}

    def do_GET(self):
        if self.path == '/api/stats':
            stats = load_stats()
            self._send_json(stats)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/question':
            body = self._read_body()
            api_key = body.get('apiKey', '')
            if not api_key:
                self._send_json({'error': '请提供 API Key'}, 400)
                return
            result = generate_question(api_key)
            if result:
                self._send_json(result)
            else:
                self._send_json({'error': 'AI 生成题目失败，请重试'}, 500)

        elif self.path == '/api/verify':
            body = self._read_body()
            expr = body.get('expression', '')
            user_ans = body.get('userAnswer', '')
            if not expr or not user_ans:
                self._send_json({'error': '缺少参数'}, 400)
                return
            result = verify_answer(expr, user_ans)
            self._send_json(result)

        elif self.path == '/api/stats':
            body = self._read_body()
            if not body:
                self._send_json({'error': '缺少数据'}, 400)
                return
            save_stats(body)
            self._send_json({'ok': True})

        else:
            self._send_json({'error': 'Not Found'}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        if '/api/' in str(args):
            return
        super().log_message(format, *args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
    print(f'''
  ╔═══════════════════════════════════════════╗
  ║   AI 口算练习 · 本地服务器                 ║
  ║                                           ║
  ║   访问地址: http://localhost:{PORT}          ║
  ║   需要配置 DeepSeek API Key 才能使用       ║
  ║                                           ║
  ║   按 Ctrl+C 停止服务器                     ║
  ╚═══════════════════════════════════════════╝
    ''')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n服务器已停止')
        server.server_close()
        sys.exit(0)
