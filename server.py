"""
AI 口算练习 - 本地服务器
启动后访问 http://localhost:3001
"""
import http.server
import json
import os
import sys
import re
import random
import urllib.request
import urllib.error
from fractions import Fraction

PORT = 3001
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


QUESTION_HISTORY_MAX = 500


def _answer_fingerprint(answer, qtype: str) -> str:
    """将答案规范化为字符串指纹，用于去重比较"""
    if qtype in ('matrix', 'inverse'):
        return json.dumps(answer)
    if qtype == 'equation':
        return json.dumps(answer, sort_keys=True)
    return str(answer).strip()


def _question_fingerprint(question_data: dict, qtype: str) -> str:
    """根据题目数据生成唯一指纹（用于矩阵/方程/逆矩阵等结构化题型）"""
    if qtype == 'matrix':
        return json.dumps({'A': question_data.get('matrixA'), 'B': question_data.get('matrixB')}, sort_keys=True)
    if qtype == 'equation':
        return json.dumps({'A': question_data.get('matrixA'), 'b': question_data.get('vectorB')}, sort_keys=True)
    if qtype == 'inverse':
        return json.dumps({'A': question_data.get('matrixA')}, sort_keys=True)
    # 口算题直接用题目文本
    return question_data.get('question', '')


def load_question_history() -> list:
    """从 stats 文件中加载题目历史"""
    stats = load_stats()
    return stats.get('questionHistory', [])


def save_question_to_history(question_data: dict, qtype: str):
    """将一道题目存入历史记录"""
    stats = load_stats()
    history = stats.get('questionHistory', [])
    entry = {
        'questionText': question_data.get('question', ''),
        'answer': _answer_fingerprint(question_data.get('answer', ''), qtype),
        'dataFingerprint': _question_fingerprint(question_data, qtype),
        'questionType': qtype,
    }
    history.append(entry)
    if len(history) > QUESTION_HISTORY_MAX:
        history = history[-QUESTION_HISTORY_MAX:]
    stats['questionHistory'] = history
    save_stats(stats)


def is_duplicate_question(question_data: dict, qtype: str, history: list) -> bool:
    """检查题目是否和历史记录中的题目重复

    去重规则：
    1. 数据指纹相同 → 重复（矩阵A和B、方程组系数等结构化数据）
    2. 答案相同 → 重复（算术题计算结果相同、矩阵乘积结果相同）
    3. 题目文本相同 → 重复（仅对算术口算题有效，结构化题型的文本是固定模板）
    """
    if not history:
        return False
    q_text = question_data.get('question', '')
    q_ans = _answer_fingerprint(question_data.get('answer', ''), qtype)
    q_fp = _question_fingerprint(question_data, qtype)
    is_structured = qtype in ('matrix', 'equation', 'inverse')
    for entry in history:
        # 数据指纹相同（结构化题型：矩阵/方程组等） → 重复
        if q_fp and entry.get('dataFingerprint') == q_fp:
            return True
        # 答案相同 → 重复
        if q_ans and entry.get('answer') == q_ans:
            return True
        # 题目文本相同（仅对算术口算题有效）
        if not is_structured and q_text and entry.get('questionText') == q_text:
            return True
    return False


def _to_frac_matrix(m):
    """将字符串矩阵（list of list of str）转为 Fraction 矩阵"""
    return [[Fraction(x) for x in row] for row in m]


def _to_frac_vector(v):
    """将字符串向量转为 Fraction 列表（兼容 1D 和 2D）"""
    if not v:
        return []
    if isinstance(v[0], list):
        # 2D 向量（逆矩阵右侧 I 矩阵）
        return [[Fraction(x) for x in row] for row in v]
    # 1D 向量（方程组右侧常数项）
    return [Fraction(x) for x in v]


def _rebuild_full_augmented(step, is_inverse):
    """将 step.augmented 重建为完整增广矩阵（Fraction）， shape 为 n × (n + right_cols)"""
    n = len(step['augmented']['matrix'])
    left = _to_frac_matrix(step['augmented']['matrix'])
    right = step['augmented']['vector']
    if is_inverse:
        # 2D: 右侧也是 n×n 矩阵
        right_mat = _to_frac_matrix(right)
        return [left[i] + right_mat[i] for i in range(n)]
    else:
        # 1D: 右侧是 n 维列向量
        right_vec = _to_frac_vector(right)
        return [left[i] + [right_vec[i]] for i in range(n)]


def verify_steps_consistency(steps):
    """验证 solutionSteps 中每步的左乘矩阵和中间结果是否一致

    检查：leftMatrix[step] × augmented[step-1] == augmented[step]
    返回 (是否全部正确, 错误信息)
    """
    if not steps or len(steps) < 2:
        return True, ''

    # 判断是否为逆矩阵题型（augmented.vector 是 2D 而非 1D）
    is_inverse = bool(steps[0].get('augmented', {}).get('vector', []) and
                      isinstance(steps[0]['augmented']['vector'][0], list))

    for i in range(1, len(steps)):
        prev = steps[i - 1]
        curr = steps[i]

        left = _to_frac_matrix(curr['leftMatrix'])
        prev_aug = _rebuild_full_augmented(prev, is_inverse)
        curr_aug = _rebuild_full_augmented(curr, is_inverse)

        n = len(left)
        # 矩阵乘法：left (n×n) × prev_aug (n×m) = result (n×m)
        m = len(prev_aug[0])
        computed = [[Fraction(0) for _ in range(m)] for _ in range(n)]
        for r in range(n):
            for c in range(m):
                for k in range(n):
                    computed[r][c] += left[r][k] * prev_aug[k][c]

        # 逐元素比较
        for r in range(n):
            for c in range(m):
                if computed[r][c] != curr_aug[r][c]:
                    return False, (f'步骤 {i+1} ("{curr["operation"]}") 验证失败: '
                                   f'位置 [{r},{c}] 期望 {curr_aug[r][c]}，计算得 {computed[r][c]}')
    return True, ''


def call_deepseek(api_key: str, prompt: str) -> str | None:
    """调用 DeepSeek API 并返回文本内容"""
    data = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096,
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


def _repair_json_fractions(text: str) -> str:
    """修复 JSON 中未用引号包裹的分数（如 -1/3 → "-1/3"）"""
    # 匹配值位置的分数 pattern: 在 : 或 [ 或 , 之后，在 , 或 ] 或 } 之前
    # 如 : -1/3  或  , 2/3  或  [-1/3,  等
    text = re.sub(
        r'(?<=[:,\[])\s*(-?\d+/\d+)\s*(?=[,\}\]])',
        r'"\1"',
        text
    )
    return text


def parse_json_from_response(text: str) -> dict | None:
    """从 AI 返回文本中提取 JSON 对象"""
    text = text.strip()

    def try_parse(t):
        if t.startswith('{') and t.endswith('}'):
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                pass
        return None

    # 尝试直接解析
    result = try_parse(text)
    if result:
        return result

    # 尝试修复分数后解析
    repaired = _repair_json_fractions(text)
    if repaired != text:
        result = try_parse(repaired)
        if result:
            return result

    # 尝试从 ```json ... ``` 中提取
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        result = try_parse(m.group(1).strip())
        if result:
            return result
        # 也尝试修复后解析
        repaired_m = _repair_json_fractions(m.group(1).strip())
        if repaired_m != m.group(1).strip():
            result = try_parse(repaired_m)
            if result:
                return result

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


# === 高斯消元 + 初等矩阵生成 ===

def format_frac(frac):
    """将 Fraction 格式化为显示字符串"""
    if frac.denominator == 1:
        return str(frac.numerator)
    return f'{frac.numerator}/{frac.denominator}'


def make_identity(n):
    """创建 n×n 单位矩阵（Fraction 类型）"""
    return [[Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]


def elementary_swap(n, i, j):
    """行交换初等矩阵：E·A 交换 A 的第 i 行和第 j 行"""
    E = make_identity(n)
    E[i][i] = E[j][j] = Fraction(0)
    E[i][j] = E[j][i] = Fraction(1)
    return E


def elementary_add(n, i, j, k):
    """行倍加初等矩阵：E·A 将 A 的第 j 行的 k 倍加到第 i 行（R_i = R_i + k·R_j）"""
    E = make_identity(n)
    E[i][j] = k
    return E


def elementary_scale(n, i, k):
    """行倍乘初等矩阵：E·A 将 A 的第 i 行乘以 k（R_i = k·R_i）"""
    E = make_identity(n)
    E[i][i] = k
    return E


def mat_to_json(m):
    """将 Fraction 矩阵转为 JSON 可序列化的字符串矩阵"""
    return [[format_frac(m[i][j]) for j in range(len(m[i]))] for i in range(len(m))]


def augmented_to_json(aug):
    """将增广矩阵 [A|b] 拆分为 matrix 和 vector 两个字符串矩阵"""
    n = len(aug)
    m = [[format_frac(aug[i][j]) for j in range(len(aug[i]) - 1)] for i in range(n)]
    v = [format_frac(aug[i][-1]) for i in range(n)]
    return {'matrix': m, 'vector': v}


def operation_text(row, col, factor):
    """生成行变换操作描述文字"""
    if factor > 0:
        factor_str = format_frac(factor)
        if factor_str == '1':
            return f'R{row+1} = R{row+1} - R{col+1}'
        return f'R{row+1} = R{row+1} - {factor_str} × R{col+1}'
    else:
        factor_str = format_frac(-factor)
        if factor_str == '1':
            return f'R{row+1} = R{row+1} + R{col+1}'
        return f'R{row+1} = R{row+1} + {factor_str} × R{col+1}'


def scale_operation_text(row, k):
    """生成行倍乘操作描述文字"""
    return f'R{row+1} = {format_frac(k)} × R{row+1}'


def gaussian_elimination_steps(A_int, b_int):
    """
    对增广矩阵 [A|b] 执行高斯-约当消元，跟踪每一步的初等矩阵。

    A_int: 整数系数矩阵（list of list of int）
    b_int: 常数向量（list of int）

    返回: { 'steps': [...], 'solution': {var: value} }
    每步: { 'operation': str, 'leftMatrix': [[str]], 'augmented': {matrix: [[str]], vector: [str]} }
    """
    n = len(A_int)

    # 转为 Fraction 精确计算
    A = [[Fraction(x) for x in row] for row in A_int]
    b = [Fraction(x) for x in b_int]
    aug = [A[i] + [b[i]] for i in range(n)]

    steps = []

    # 第 0 步：初始状态
    steps.append({
        'operation': '写出增广矩阵',
        'leftMatrix': mat_to_json(make_identity(n)),
        'augmented': augmented_to_json(aug),
    })

    # 前向消元（化为行阶梯形）
    for col in range(n):
        # 寻找主元
        pivot_row = None
        for row in range(col, n):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue  # 奇异矩阵，跳过（但 AI 已验证解存在）

        # 交换行（如需）
        if pivot_row != col:
            E = elementary_swap(n, col, pivot_row)
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            steps.append({
                'operation': f'交换 R{col+1} 和 R{pivot_row+1}',
                'leftMatrix': mat_to_json(E),
                'augmented': augmented_to_json(aug),
            })

        # 消去下方元素
        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            if factor != 0:
                E = elementary_add(n, row, col, -factor)
                for k in range(col, n + 1):
                    aug[row][k] -= factor * aug[col][k]
                steps.append({
                    'operation': operation_text(row, col, factor),
                    'leftMatrix': mat_to_json(E),
                    'augmented': augmented_to_json(aug),
                })

    # 后向消元（化为行最简形 RREF）
    for col in range(n - 1, -1, -1):
        pivot = aug[col][col]
        # 归一化主元为 1
        if pivot != 1 and pivot != 0:
            inv = Fraction(1) / pivot
            E = elementary_scale(n, col, inv)
            for k in range(n + 1):
                aug[col][k] *= inv
            steps.append({
                'operation': scale_operation_text(col, inv),
                'leftMatrix': mat_to_json(E),
                'augmented': augmented_to_json(aug),
            })

        # 消去上方元素
        for row in range(col):
            factor = aug[row][col]
            if factor != 0:
                E = elementary_add(n, row, col, -factor)
                for k in range(n + 1):
                    aug[row][k] -= factor * aug[col][k]
                steps.append({
                    'operation': operation_text(row, col, factor),
                    'leftMatrix': mat_to_json(E),
                    'augmented': augmented_to_json(aug),
                })

    # 提取解向量（使用 format_frac 处理可能的非整数解）
    solution = {f'x{i+1}': format_frac(aug[i][-1]) for i in range(n)}

    return {'steps': steps, 'solution': solution}


def gaussian_elimination_inverse_steps(A_int):
    """
    对增广矩阵 [A|I] 执行高斯-约当消元求逆矩阵，跟踪每一步的初等矩阵。

    A_int: n×n 整数矩阵（list of list of int）

    返回: { 'steps': [...], 'inverse': [[str]] }
    每步: { 'operation': str, 'leftMatrix': [[str]], 'augmented': {matrix: [[str]], vector: [[str]]} }
    其中 augmented.vector 是右侧 I 矩阵的当前状态（按列展开为 list of list）
    """
    n = len(A_int)

    # 转为 Fraction 精确计算
    A = [[Fraction(x) for x in row] for row in A_int]
    # 增广矩阵 [A | I]：n 行, 2n 列
    aug = [A[i] + [Fraction(1) if j == i else Fraction(0) for j in range(n)] for i in range(n)]

    steps = []

    # 第 0 步：初始状态 [A | I]
    steps.append({
        'operation': '写出增广矩阵 [A | I]',
        'leftMatrix': mat_to_json(make_identity(n)),
        'augmented': augmented_to_json_inv(aug, n),
    })

    # 前向消元（化为行阶梯形）
    for col in range(n):
        pivot_row = None
        for row in range(col, n):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            raise ValueError(f'矩阵奇异，第 {col+1} 列无法找到主元')

        # 交换行
        if pivot_row != col:
            E = elementary_swap(n, col, pivot_row)
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            steps.append({
                'operation': f'交换 R{col+1} 和 R{pivot_row+1}',
                'leftMatrix': mat_to_json(E),
                'augmented': augmented_to_json_inv(aug, n),
            })

        # 消去下方元素
        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            if factor != 0:
                E = elementary_add(n, row, col, -factor)
                for k in range(2 * n):
                    aug[row][k] -= factor * aug[col][k]
                steps.append({
                    'operation': operation_text(row, col, factor),
                    'leftMatrix': mat_to_json(E),
                    'augmented': augmented_to_json_inv(aug, n),
                })

    # 后向消元（化为行最简形 RREF）
    for col in range(n - 1, -1, -1):
        pivot = aug[col][col]
        # 归一化主元为 1
        if pivot != 1 and pivot != 0:
            inv = Fraction(1) / pivot
            E = elementary_scale(n, col, inv)
            for k in range(2 * n):
                aug[col][k] *= inv
            steps.append({
                'operation': scale_operation_text(col, inv),
                'leftMatrix': mat_to_json(E),
                'augmented': augmented_to_json_inv(aug, n),
            })

        # 消去上方元素
        for row in range(col):
            factor = aug[row][col]
            if factor != 0:
                E = elementary_add(n, row, col, -factor)
                for k in range(2 * n):
                    aug[row][k] -= factor * aug[col][k]
                steps.append({
                    'operation': operation_text(row, col, factor),
                    'leftMatrix': mat_to_json(E),
                    'augmented': augmented_to_json_inv(aug, n),
                })

    # 提取逆矩阵（右侧 n 列）
    inverse = [[format_frac(aug[i][n + j]) for j in range(n)] for i in range(n)]

    return {'steps': steps, 'inverse': inverse}


def augmented_to_json_inv(aug, n):
    """将 [A|I] 增广矩阵拆分为 matrix 和 vector（右侧是方阵，用二维数组表示）"""
    m = [[format_frac(aug[i][j]) for j in range(n)] for i in range(n)]
    # 右侧 I 部分也是 n×n 矩阵
    v = [[format_frac(aug[i][n + j]) for j in range(n)] for i in range(n)]
    return {'matrix': m, 'vector': v}


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


MATRIX_PROMPT = """你是一个矩阵乘法出题器。请生成一道矩阵乘法题。

约束：
- 矩阵A的维度为 m×n，矩阵B的维度为 n×l
- m, n, l 各自在 1 到 3 之间
- 矩阵中的每个数字都是 -10 到 10 之间的一位整数，不允许小数或分数
- 结果矩阵的每个元素不能超过3位数字

请严格按以下 JSON 格式返回，不要包含其他内容：
{
  "question": "计算矩阵 A × B",
  "matrixA": [[每行用逗号分隔]],
  "matrixB": [[每行用逗号分隔]],
  "answer": [[计算结果的每一行]],
  "rows": m,
  "cols": l,
  "innerDim": n,
  "solution": "详细的解题步骤，每步单独一行"
}"""

DECIMAL_PROMPT = """你是一个小数计算题出题器。请生成一道**必须包含小数**的口算题。

数字约束：
- 题目中必须包含小数（如 1.5, 3.14, 0.2），不能有分数
- 每个数字最多2位（整数部分和小数部分分别不超过2位）
- 例如允许：5, 1.5, 23.45, 0.5
- 例如不允许：123, 1.234, 32/3

运算约束：
- 最多2个加法(+)、2个减法(-)、2个乘法(×)、2个除法(÷)
- 可以使用括号
- 运算符总数不超过8个

结果约束：
- 计算结果必须在 -1000 到 1000 之间
- 结果必须是小数（或整数），不能用分数表示

类型一致性（重要！）：
- 答案必须用小数表示（如果结果是整数也用整数）
- 不能出现分数

请严格按以下 JSON 格式返回，不要包含其他内容：
{
  "question": "题目文本（用 × 和 ÷ 符号）",
  "expression": "可用于 Python eval 的表达式（用 * 和 /）",
  "answer": "标准答案（小数，如 23.456）",
  "answerType": "decimal",
  "solution": "详细的解题步骤，用中文，分步说明，每步单独一行"
}"""

FRACTION_PROMPT = """你是一个分数计算题出题器。请生成一道**必须包含分数**的口算题。

数字约束：
- 题目中必须包含分数（如 2/3, 1/2, 7/5），不能有小数
- 每个数字最多2位（分子分母各自不超过2位）
- 例如允许：5, 2/3, 32/63, 17/3
- 例如不允许：123, 1.5, 123/456

运算约束：
- 最多2个加法(+)、2个减法(-)、2个乘法(×)、2个除法(÷)
- 可以使用括号
- 运算符总数不超过8个

结果约束：
- 计算结果必须在 -1000 到 1000 之间
- 结果必须是分数（或整数），不能用小数表示
- 分数必须化为最简形式（如 1/2 而非 2/4）

类型一致性（重要！）：
- 答案必须用最简分数表示
- 不能出现小数

请严格按以下 JSON 格式返回，不要包含其他内容：
{
  "question": "题目文本（用 × 和 ÷ 符号）",
  "expression": "可用于 Python eval 的表达式（用 * 和 /）",
  "answer": "标准答案（最简分数，如 697/63）",
  "answerType": "fraction",
  "solution": "详细的解题步骤，用中文，分步说明，每步单独一行"
}"""

EQUATION_PROMPT = """你是一个线性方程组出题器。请生成一道{n}元线性方程组求解题。

请按以下步骤设计：
1. 确定解向量：选择 {n} 个整数 x1, x2, ..., x{n}，每个在 -5 到 5 之间
2. 构造系数矩阵 A：一个 {n}×{n} 矩阵，每个元素为 -5 到 5 之间的整数
3. 矩阵 A 必须可逆（行列式不为零）

注意事项：
- 解向量各元素是小的整数（-5 到 5）
- 系数矩阵各元素也是小的整数（-5 到 5）
- 等号右侧的常数项由系统自动计算，你不需要提供

请严格按以下 JSON 格式返回，不要包含其他内容：
{{
  "question": "解下列线性方程组",
  "equations": ["方程1的文本", "方程2的文本" {extra_eq}],
  "matrixA": [[系数矩阵]],
  "variables": {variables_list},
  "numVars": {n},
  "answer": {answer_example},
  "solution": "详细的解题步骤，使用行变换方法，每步单独一行，包含变换操作说明"
}}"""


INVERSE_MATRIX_PROMPT = """你是一个矩阵求逆出题器。请生成一道{n}×{n}矩阵求逆题。

约束：
- 生成一个 {n}×{n} 方阵 A
- 矩阵中的每个数字都是 -5 到 5 之间的整数（不要小数或分数）
- 矩阵必须可逆（行列式不为零）
- 逆矩阵的元素可能是分数

重要——JSON 格式要求：
- "answer" 字段中的矩阵元素，如果是分数必须用**引号包裹成字符串**（如 "1/3"），不能写成 1/3
- 整数不用引号（如 2）
- 例如正确答案是 [[1/3, 0], [0, 1/2]] 时，应写成 [["1/3", 0], [0, "1/2"]]

请严格按以下 JSON 格式返回，不要包含其他内容：
{{
  "question": "求矩阵 A 的逆矩阵",
  "matrixA": [[矩阵的每一行]],
  "size": {n},
  "answer": [[逆矩阵的每一行（分数用引号包裹）]],
  "solution": "详细的解题步骤，使用行变换方法 [A|I] → [I|A^(-1)]，每步单独一行"
}}"""


def generate_question(api_key: str, question_type: str = 'mixed') -> dict | None:
    """生成一道符合约束的题目，最多重试3次

    question_type 取值:
      'mixed'         — 混合口算（整数/小数/分数均可）
      'decimal'       — 小数计算（必须包含小数，不能有分数）
      'fraction'      — 分数计算（必须包含分数，不能有小数）
      'matrix'        — 矩阵乘法
      'equation'      — 线性方程组
      'inverse'       — 逆矩阵
    """
    if question_type == 'matrix':
        prompt = MATRIX_PROMPT
    elif question_type == 'decimal':
        prompt = DECIMAL_PROMPT
    elif question_type == 'fraction':
        prompt = FRACTION_PROMPT
    elif question_type == 'equation':
        n = random.choice([2, 3])
        print(f'[equation] 随机选择维度: {n}元')
        variables_list = json.dumps(['x', 'y'] if n == 2 else ['x', 'y', 'z'])
        answer_example = json.dumps({'x': 1, 'y': 2} if n == 2 else {'x': 1, 'y': 2, 'z': 3})
        extra_eq = '' if n == 2 else ', "方程3的文本"'
        prompt = EQUATION_PROMPT.format(n=n, extra_eq=extra_eq, variables_list=variables_list, answer_example=answer_example)
    elif question_type == 'inverse':
        n = random.choice([2, 3])
        print(f'[inverse] 随机选择维度: {n}×{n}')
        prompt = INVERSE_MATRIX_PROMPT.format(n=n)
    else:
        prompt = SYSTEM_PROMPT
    max_attempts = 3
    if question_type in ('equation', 'inverse', 'matrix'):
        max_attempts = 5
    for attempt in range(max_attempts):
        print(f'[{question_type}] 尝试生成第 {attempt + 1} 次...')
        raw = call_deepseek(api_key, prompt)
        if not raw:
            continue

        data = parse_json_from_response(raw)
        if not data:
            print(f'[{question_type}] 无法解析 AI 响应（前200字符）: {raw[:200]}')
            continue

        if question_type == 'matrix':
            question = data.get('question', '').strip()
            matrix_a = data.get('matrixA')
            matrix_b = data.get('matrixB')
            answer = data.get('answer')
            rows = data.get('rows', 0)
            cols = data.get('cols', 0)
            inner_dim = data.get('innerDim', 0)
            solution = data.get('solution', '').strip()

            if not all([question, matrix_a, matrix_b, answer, rows, cols, inner_dim, solution]):
                print(f'[matrix] 字段不完整，跳过')
                continue

            # 验证维度
            if not (1 <= rows <= 3 and 1 <= cols <= 3 and 1 <= inner_dim <= 3):
                print(f'[matrix] 维度超出范围，跳过')
                continue

            if len(matrix_a) != rows or any(len(r) != inner_dim for r in matrix_a):
                print(f'[matrix] 矩阵A维度不匹配，跳过')
                continue
            if len(matrix_b) != inner_dim or any(len(r) != cols for r in matrix_b):
                print(f'[matrix] 矩阵B维度不匹配，跳过')
                continue
            if len(answer) != rows or any(len(r) != cols for r in answer):
                print(f'[matrix] 结果矩阵维度不匹配，跳过')
                continue

            # 验证所有数字在 -10 到 10 之间
            val_ok = True
            for mtx in (matrix_a, matrix_b):
                for row in mtx:
                    for val in row:
                        if not isinstance(val, int) or val < -10 or val > 10:
                            print(f'[matrix] 包含范围外的数字 {val}，跳过')
                            val_ok = False
                            break
                    if not val_ok:
                        break
                if not val_ok:
                    break
            if not val_ok:
                continue

            # 计算实际乘积验证 AI 结果
            computed = [[sum(matrix_a[i][k] * matrix_b[k][j] for k in range(inner_dim)) for j in range(cols)] for i in range(rows)]
            if computed != answer:
                print(f'[matrix] AI计算结果有误，使用服务端矫正')
                answer = computed

            # 验证结果每个元素不超过3位
            for row in answer:
                for val in row:
                    if val > 999 or val < -999:
                        print(f'[matrix] 结果元素超标，跳过')
                        continue

            print(f'[matrix] 生成成功: {rows}×{inner_dim} · {inner_dim}×{cols}')
            # 去重检查
            question_data = {'question': question, 'answer': answer, 'matrixA': matrix_a, 'matrixB': matrix_b}
            history = load_question_history()
            if is_duplicate_question(question_data, 'matrix', history):
                print(f'[matrix] 与历史记录重复，重新生成')
                continue
            save_question_to_history(question_data, 'matrix')
            return {
                'questionType': 'matrix',
                'question': question,
                'matrixA': matrix_a,
                'matrixB': matrix_b,
                'answer': answer,
                'answerType': 'matrix',
                'rows': rows,
                'cols': cols,
                'innerDim': inner_dim,
                'solution': solution,
            }

        if question_type == 'equation':
            equations = data.get('equations', [])
            matrix_a = data.get('matrixA')
            variables = data.get('variables', [])
            num_vars = data.get('numVars', 0)
            answer = data.get('answer', {})
            solution = data.get('solution', '').strip()

            if not all([equations, matrix_a, variables, num_vars, answer, solution]):
                print(f'[equation] 字段不完整（equations/matrixA/variables/numVars/answer/solution），跳过')
                continue

            if num_vars not in (2, 3):
                print(f'[equation] 元数 {num_vars} 不合法，跳过')
                continue

            if len(equations) != num_vars or len(matrix_a) != num_vars:
                print(f'[equation] 方程数量不匹配，跳过')
                continue

            # 验证系数矩阵元素在 -5 到 5 之间
            val_ok = True
            for row in matrix_a:
                if len(row) != num_vars:
                    print(f'[equation] 系数矩阵列数不匹配，跳过')
                    val_ok = False
                    break
                for val in row:
                    if not isinstance(val, int) or val < -5 or val > 5:
                        print(f'[equation] 系数 {val} 超出范围（-5~5），跳过')
                        val_ok = False
                        break
                if not val_ok:
                    break
            if not val_ok:
                continue

            # 验证答案（解向量）为整数且在 -5 到 5 之间
            val_ok = True
            for var in variables:
                val = answer.get(var)
                if val is None or not isinstance(val, int) or val < -5 or val > 5:
                    print(f'[equation] 解 {var}={val} 超出范围（-5~5），跳过')
                    val_ok = False
                    break
            if not val_ok:
                continue
            if len(answer) != num_vars:
                print(f'[equation] 解向量长度不匹配，跳过')
                continue

            # 由服务器计算 b = A × solution（保证数学正确）
            var_list = variables
            n = num_vars
            vector_b = [sum(matrix_a[i][j] * answer[var_list[j]] for j in range(n)) for i in range(n)]

            # 验证常数项在合理范围内
            val_ok = True
            for val in vector_b:
                if val < -100 or val > 100:
                    print(f'[equation] 常数项 {val} 超出范围（-100~100），跳过')
                    val_ok = False
                    break
            if not val_ok:
                continue

            print(f'[equation] 验证通过，使用服务端高斯消元生成步骤')

            # 使用服务端高斯消元生成解题步骤
            try:
                elim_result = gaussian_elimination_steps(matrix_a, vector_b)
                solution_steps = elim_result['steps']
                # 验证每步的左乘矩阵与中间结果：左乘矩阵 × 上一增广矩阵 = 当前增广矩阵
                steps_ok, err_msg = verify_steps_consistency(solution_steps)
                if not steps_ok:
                    print(f'[equation] {err_msg}，跳过')
                    continue
            except Exception as e:
                print(f'[equation] 高斯消元出错: {e}，跳过', file=sys.stderr)
                continue

            # 去重检查
            question_data = {'question': '解下列线性方程组', 'answer': answer, 'matrixA': matrix_a, 'vectorB': vector_b}
            history = load_question_history()
            if is_duplicate_question(question_data, 'equation', history):
                print(f'[equation] 与历史记录重复，重新生成')
                continue
            save_question_to_history(question_data, 'equation')
            return {
                'questionType': 'equation',
                'question': '解下列线性方程组',
                'equations': equations,
                'matrixA': matrix_a,
                'vectorB': vector_b,
                'variables': variables,
                'numVars': num_vars,
                'answer': answer,
                'answerType': 'equation',
                'solution': solution,
                'solutionSteps': solution_steps,
            }

        if question_type == 'inverse':
            matrix_a = data.get('matrixA')
            size = data.get('size', 0)
            answer = data.get('answer')
            solution = data.get('solution', '').strip()

            if not all([matrix_a, size, answer, solution]):
                print(f'[inverse] 字段不完整，跳过')
                continue

            if size not in (2, 3):
                print(f'[inverse] 矩阵尺寸 {size} 不合法，跳过')
                continue

            if len(matrix_a) != size or any(len(r) != size for r in matrix_a):
                print(f'[inverse] 矩阵维度不匹配，跳过')
                continue

            # 验证所有数字在 -5 到 5 之间
            val_ok = True
            for row in matrix_a:
                for val in row:
                    if not isinstance(val, int) or val < -5 or val > 5:
                        print(f'[inverse] 元素 {val} 超出范围，跳过')
                        val_ok = False
                        break
                if not val_ok:
                    break
            if not val_ok:
                continue

            # 验证 AI 给出的逆矩阵是否正确：A × A_inv = I（支持分数字符串）
            n = size
            try:
                answer_frac = []
                for row in answer:
                    answer_frac.append([Fraction(x) for x in row])
                computed_inv_prod = [[sum(matrix_a[i][k] * answer_frac[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
                is_identity = all(
                    abs(computed_inv_prod[i][j] - (1 if i == j else 0)) < Fraction(1, 1000000)
                    for i in range(n) for j in range(n)
                )
            except Exception as e:
                print(f'[inverse] AI 逆矩阵验证出错: {e}，跳过')
                continue
            if not is_identity:
                print(f'[inverse] AI 给出的逆矩阵不正确，使用服务端计算')
                # 不跳过，让服务端矫正

            print(f'[inverse] 生成成功: {size}×{size}')

            # 使用服务端高斯消元生成解题步骤（包含左乘矩阵）
            try:
                inv_result = gaussian_elimination_inverse_steps(matrix_a)
                solution_steps = inv_result['steps']
                server_inverse = inv_result['inverse']
                # 验证每步的左乘矩阵与中间结果
                steps_ok, err_msg = verify_steps_consistency(solution_steps)
                if not steps_ok:
                    print(f'[inverse] {err_msg}，跳过')
                    continue
            except Exception as e:
                print(f'[inverse] 高斯消元求逆出错: {e}，回退到 AI 生成的文本', file=sys.stderr)
                solution_steps = None
                server_inverse = None

            # 使用服务端计算的逆矩阵作为正确答案（更可靠）
            final_answer = server_inverse if server_inverse else answer

            # 去重检查
            question_data = {'question': '求矩阵 A 的逆矩阵', 'answer': final_answer, 'matrixA': matrix_a}
            history = load_question_history()
            if is_duplicate_question(question_data, 'inverse', history):
                print(f'[inverse] 与历史记录重复，重新生成')
                continue
            save_question_to_history(question_data, 'inverse')
            return {
                'questionType': 'inverse',
                'question': f'求矩阵 A 的逆矩阵',
                'matrixA': matrix_a,
                'size': size,
                'answer': final_answer,
                'answerType': 'inverse',
                'solution': solution,
                'solutionSteps': solution_steps,
            }

        # 原有的口算题逻辑
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
        # 去重检查
        question_data = {'question': question, 'answer': server_answer}
        history = load_question_history()
        if is_duplicate_question(question_data, 'arithmetic', history):
            print(f'[Question] 与历史记录重复，重新生成')
            continue
        save_question_to_history(question_data, 'arithmetic')
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


def verify_matrix_answer(matrix_a: list, matrix_b: list, user_answer: list) -> dict:
    """核验矩阵乘法答案"""
    try:
        rows = len(matrix_a)
        inner_dim = len(matrix_a[0]) if rows else 0
        cols = len(matrix_b[0]) if matrix_b else 0

        # 计算正确结果
        computed = [[sum(matrix_a[i][k] * matrix_b[k][j] for k in range(inner_dim)) for j in range(cols)] for i in range(rows)]

        # 比较
        if len(user_answer) != rows:
            return {'correct': False, 'error': '行数不匹配', 'exactAnswer': computed}
        correct = True
        for i in range(rows):
            if len(user_answer[i]) != cols:
                return {'correct': False, 'error': '列数不匹配', 'exactAnswer': computed}
            for j in range(cols):
                # 将用户输入转为 int 再比较（前端可能发字符串或数字）
                try:
                    user_val = int(user_answer[i][j]) if not isinstance(user_answer[i][j], int) else user_answer[i][j]
                except (ValueError, TypeError):
                    correct = False
                    continue
                if user_val != computed[i][j]:
                    correct = False

        return {'correct': correct, 'exactAnswer': computed}
    except (IndexError, TypeError, ValueError):
        return {'correct': False, 'error': '矩阵格式错误'}


def verify_equation_answer(correct_answer: dict, user_answer: dict, variables: list) -> dict:
    """核验方程组答案"""
    try:
        correct = True
        for var in variables:
            user_val = user_answer.get(var)
            if user_val is None:
                return {'correct': False, 'error': f'缺少 {var} 的值'}
            try:
                user_val = int(user_val) if not isinstance(user_val, int) else user_val
            except (ValueError, TypeError):
                correct = False
                continue
            if user_val != correct_answer.get(var):
                correct = False

        return {'correct': correct, 'exactAnswer': correct_answer}
    except (TypeError, ValueError):
        return {'correct': False, 'error': '答案格式错误'}


def verify_inverse_answer(matrix_a: list, user_answer: list, size: int) -> dict:
    """核验逆矩阵答案：验证 A × user_inv ≈ I"""
    try:
        if len(user_answer) != size:
            return {'correct': False, 'error': '行数不匹配'}
        for row in user_answer:
            if len(row) != size:
                return {'correct': False, 'error': '列数不匹配'}

        # 将用户输入解析为 Fraction
        user_frac = []
        for i in range(size):
            row_frac = []
            for j in range(size):
                val = user_answer[i][j]
                if isinstance(val, str):
                    if '/' in val:
                        parts = val.split('/')
                        row_frac.append(Fraction(int(parts[0]), int(parts[1])))
                    else:
                        row_frac.append(Fraction(int(val)))
                else:
                    row_frac.append(Fraction(int(val)))
            user_frac.append(row_frac)

        # 计算 A × user_inv
        prod = [[sum(matrix_a[i][k] * user_frac[k][j] for k in range(size)) for j in range(size)] for i in range(size)]

        # 检查结果是否接近单位矩阵（允许分数误差）
        correct = True
        for i in range(size):
            for j in range(size):
                expected = Fraction(1) if i == j else Fraction(0)
                diff = abs(prod[i][j] - expected)
                if diff > Fraction(1, 1000):
                    correct = False

        return {'correct': correct, 'exactAnswer': None}
    except (IndexError, TypeError, ValueError, ZeroDivisionError) as e:
        return {'correct': False, 'error': f'格式错误: {e}'}


class Handler(http.server.SimpleHTTPRequestHandler):

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _send_file(self, path: str, content_type: str):
        """发送本地文件，带防缓存头"""
        try:
            with open(path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            self.wfile.write(data)
        except IOError:
            self.send_error(404, 'File not found')

    def do_GET(self):
        if self.path == '/api/ping':
            self._send_json({'ok': True})
        elif self.path == '/api/stats':
            stats = load_stats()
            self._send_json(stats)
        elif self.path == '/api/records':
            stats = load_stats()
            records = stats.get('records', [])
            self._send_json({'records': records})
        elif self.path.split('?')[0] in ('/', '/index.html'):
            self._send_file(os.path.join(os.path.dirname(__file__), 'index.html'), 'text/html; charset=utf-8')
        else:
            super().do_GET()

    def _read_body(self) -> dict:
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8')
            return json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            return {}

    def do_POST(self):
        if self.path == '/api/question':
            body = self._read_body()
            api_key = body.get('apiKey', '')
            if not api_key:
                self._send_json({'error': '请提供 API Key'}, 400)
                return
            qtype = body.get('questionType', 'arithmetic')
            result = generate_question(api_key, qtype)
            if result:
                self._send_json(result)
            else:
                self._send_json({'error': 'AI 生成题目失败，请重试'}, 500)

        elif self.path == '/api/verify':
            body = self._read_body()
            qtype = body.get('questionType', 'arithmetic')

            if qtype == 'matrix':
                matrix_a = body.get('matrixA')
                matrix_b = body.get('matrixB')
                user_ans = body.get('userAnswer')
                if not all([matrix_a, matrix_b, user_ans]):
                    self._send_json({'error': '缺少矩阵参数'}, 400)
                    return
                result = verify_matrix_answer(matrix_a, matrix_b, user_ans)
                self._send_json(result)
            elif qtype == 'equation':
                correct_ans = body.get('correctAnswer', {})
                user_ans = body.get('userAnswer', {})
                variables = body.get('variables', [])
                if not correct_ans or not user_ans or not variables:
                    self._send_json({'error': '缺少参数'}, 400)
                    return
                result = verify_equation_answer(correct_ans, user_ans, variables)
                self._send_json(result)
            elif qtype == 'inverse':
                matrix_a = body.get('matrixA')
                user_ans = body.get('userAnswer')
                size = body.get('size', 0)
                if not matrix_a or not user_ans or not size:
                    self._send_json({'error': '缺少参数'}, 400)
                    return
                result = verify_inverse_answer(matrix_a, user_ans, size)
                self._send_json(result)
            else:
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

        elif self.path == '/api/records':
            body = self._read_body()
            if not body or 'records' not in body:
                self._send_json({'error': '缺少 records 字段'}, 400)
                return
            stats = load_stats()
            stats['records'] = body['records']
            save_stats(stats)
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
    try:
        server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
    except OSError as e:
        if 'Address already in use' in str(e) or '10048' in str(e):
            print(f'\n  ❌ 端口 {PORT} 已被占用！')
            print(f'     可能已有旧服务器在运行。')
            print(f'     请先关闭其他程序，或重启电脑后再试。\n')
        else:
            print(f'\n  ❌ 启动失败: {e}\n')
        sys.exit(1)
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
