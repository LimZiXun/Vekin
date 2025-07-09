
#!/usr/bin/env python3
"""
Agentic Automatic Testing Tool
Automatically generates test cases to achieve 100% code coverage using LLM
"""

import ast
import coverage
import subprocess
import sys
import os
from typing import List, Dict
import google.generativeai as genai
import re
from dotenv import load_dotenv
load_dotenv()

# Configuration
TARGET_MODULE = "triangle.py"
TEST_FILE = "Triangle_test.py"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash"

class CodeAnalyzer:
    """Analyzes Python code to extract functions and branches"""

    def __init__(self, target_file: str):
        self.target_file = target_file

    def extract_functions(self) -> List[Dict]:
        """Extract all functions from the target module"""
        try:
            with open(self.target_file, 'r') as f:
                tree = ast.parse(f.read())
        except FileNotFoundError:
            print(f"❌ Error: {self.target_file} not found")
            return []
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'lineno': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'body_lines': list(range(node.lineno, node.end_lineno + 1))
                }
                functions.append(func_info)
        return functions

    def extract_branches(self) -> List[Dict]:
        """Extract conditional branches (if statements) from the code"""
        try:
            with open(self.target_file, 'r') as f:
                tree = ast.parse(f.read())
        except FileNotFoundError:
            print(f"❌ Error: {self.target_file} not found")
            return []
        branches = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                branch_info = {
                    'lineno': node.lineno,
                    'condition': ast.unparse(node.test) if hasattr(ast, 'unparse') else '',
                    'has_else': node.orelse is not None
                }
                branches.append(branch_info)
        return branches

class CoverageAnalyzer:
    """Analyzes test coverage and identifies missing coverage"""

    def __init__(self, target_module: str, test_file: str):
        self.target_module = target_module
        self.test_file = test_file
        self.cov = None

    def run_coverage_analysis(self) -> Dict:
        """Run coverage analysis and return coverage report"""
        module_name_for_source = self.target_module.replace('.py', '')
        self.cov = coverage.Coverage(source=[module_name_for_source], branch=True)
        try:
            command = [
                sys.executable, '-m', 'coverage', 'run', '--branch',
                f'--source={module_name_for_source}',
                '-m', 'pytest', self.test_file, '-v'
            ]
            result = subprocess.run(command, capture_output=True, text=True, cwd='.')
            if result.returncode != 0:
                print(f"⚠️ Coverage run returned non-zero exit code: {result.returncode}")
                print(f"STDERR: {result.stderr}")
            self.cov.load()
        except Exception as e:
            print(f"❌ Error running coverage analysis: {e}")
            return {
                'filename': self.target_module,
                'total_statements': 0,
                'covered_statements': 0,
                'missing_lines': [],
                'missing_branches': [],
                'coverage_percentage': 0,
                'branch_coverage_percentage': 0,
                'is_100_percent': False
            }
        return self._generate_coverage_report(module_name_for_source)

    def _generate_coverage_report(self, module_name: str) -> Dict:
        """Generate detailed coverage report"""
        try:
            analysis = self.cov.analysis2(self.target_module)
            filename, statements, excluded, missing, missing_branches = analysis
            total_statements = len(statements)
            covered_statements = total_statements - len(missing)
            coverage_percentage = (covered_statements / total_statements * 100) if total_statements > 0 else 100

            coverage_data = self.cov.get_data()
            total_branches = len(coverage_data.arcs(filename)) if filename in coverage_data.measured_files() else 0
            executed_branches = len(coverage_data.executed_arcs(filename)) if filename in coverage_data.measured_files() else 0
            branch_coverage_percentage = (executed_branches / total_branches * 100) if total_branches > 0 else 100

            return {
                'filename': filename,
                'total_statements': total_statements,
                'covered_statements': covered_statements,
                'missing_lines': missing,
                'missing_branches': missing_branches,
                'coverage_percentage': coverage_percentage,
                'branch_coverage_percentage': branch_coverage_percentage,
                'is_100_percent': coverage_percentage >= 100.0 and branch_coverage_percentage >= 100.0
            }
        except Exception as e:
            print(f"Error generating coverage report: {e}")
            return {
                'filename': self.target_module,
                'total_statements': 0,
                'covered_statements': 0,
                'missing_lines': [],
                'missing_branches': [],
                'coverage_percentage': 0,
                'branch_coverage_percentage': 0,
                'is_100_percent': False
            }

class GeminiTestGenerator:
    """Uses Gemini API to generate missing test cases"""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Gemini API"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✅ Gemini API initialized successfully with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing Gemini API: {e}")
            print("❌ Please set your API key in a .env file or check your internet connection")
            self.model = None

    def generate_test_cases(self, target_code: str, existing_tests: str, missing_lines: List[int], missing_branches: List, functions: List[Dict]) -> str:
        """Generate test cases to cover missing lines and branches using Gemini"""
        if self.model is None:
            print("🔄 Gemini API not available, using rule-based generation...")
            return self._generate_rule_based_tests(target_code, missing_lines, missing_branches, functions)
        prompt = self._create_gemini_prompt(target_code, existing_tests, missing_lines, missing_branches, functions)
        try:
            print("🤖 Generating test cases with Gemini...")
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 2000,
            }
            response = self.model.generate_content(prompt, generation_config=generation_config)
            if response.text:
                generated_tests = self._extract_test_code(response.text)
                if self._validate_generated_code(generated_tests):
                    print("✅ Test cases generated successfully")
                    return generated_tests
                else:
                    print("⚠️ Generated code validation failed, using fallback")
                    return self._generate_rule_based_tests(target_code, missing_lines, missing_branches, functions)
            else:
                print("⚠️ Empty response from Gemini, using fallback")
                return self._generate_rule_based_tests(target_code, missing_lines, missing_branches, functions)
        except Exception as e:
            print(f"❌ Error generating tests with Gemini: {e}")
            print("🔄 Falling back to rule-based test generation")
            return self._generate_rule_based_tests(target_code, missing_lines, missing_branches, functions)

    def _create_gemini_prompt(self, target_code: str, existing_tests: str, missing_lines: List[int], missing_branches: List, functions: List[Dict]) -> str:
        """Create a detailed prompt for Gemini API"""
        missing_analysis = self._analyze_missing_coverage(target_code, missing_lines, missing_branches)
        prompt = f"""You are an expert Python test engineer. Generate a complete pytest test file (`Triangle_test.py`) to achieve 100% code and branch coverage for the provided `triangle.py`, replacing any existing tests to ensure correctness and proper syntax.

**Target Code to Test (`triangle.py`):**
```python
{target_code}
```

**Existing Test File (`Triangle_test.py`):**
```python
{existing_tests}
```

**Coverage Analysis:**
- Missing lines: {missing_lines}
- Missing branches: {missing_branches}
- Functions: {[f['name'] for f in functions]}
- Missing coverage analysis: {missing_analysis}

**Requirements:**
1. Generate a complete test file with:
   - Imports: `pytest`, `triangle_type`, `is_valid_triangle`
   - Fixture: `capture_output` using `@pytest.fixture` and `capsys`
   - Tests for `is_valid_triangle` and `triangle_type`
2. Use `@pytest.mark.parametrize` for all test cases
3. Ensure 100% coverage, testing all branches:
   - `is_valid_triangle`: `a > 0`, `b > 0`, `c > 0`, `a + b > c`, `b + c > a`, `c + a > b`
   - `triangle_type`: "Invalid", "Equilateral", "Isosceles", "Scalene"
4. Include edge cases:
   - Large numbers (e.g., 1e10)
   - Small positive numbers (e.g., 1e-6)
   - Floating-point inputs (e.g., 2.5, 0.1)
   - Boundary cases (e.g., a + b = c, a + b slightly > c)
   - Zero and negative sides
5. Ensure syntactically correct Python:
   - Close all brackets ([], (), {{}})
   - Use `@pytest.fixture` only for `capture_output`
   - Avoid invalid decorators on test functions
6. Test cases must be accurate:
   - `is_valid_triangle`: False for `(0.1, 0.1, 0.2)`, `(100, 1, 101)` (triangle inequality)
   - `triangle_type`: "Invalid" for invalid triangles
7. Avoid duplicate test cases
8. Return ONLY Python code, no explanations
9. Use format:
```python
@pytest.mark.parametrize("a,b,c,expected", [(x,y,z,"result"), ...])
def test_name(a, b, c, expected, capture_output):
    actual = function_name(a, b, c)
    capture_output(actual)
    assert actual == expected
```
10. Ensure the file is ready to run without modifications

Generate the complete `Triangle_test.py`:
```python
import pytest
from triangle import triangle_type, is_valid_triangle

@pytest.fixture
def capture_output(capsys):
    def _capture(actual):
        print(f"Actual Output: {{actual}}")
        captured = capsys.readouterr()
        return actual
    return _capture
```"""
        return prompt

    def _analyze_missing_coverage(self, target_code: str, missing_lines: List[int], missing_branches: List) -> str:
        """Analyze which parts of the code are missing coverage"""
        lines = target_code.splitlines()
        missing_analysis = []
        for line_num in missing_lines:
            if line_num <= len(lines):
                line_content = lines[line_num - 1].strip()
                if 'Equilateral' in line_content:
                    missing_analysis.append("Equilateral triangle case not tested")
                elif 'Isosceles' in line_content:
                    missing_analysis.append("Some isosceles triangle cases not tested")
                elif 'Scalene' in line_content:
                    missing_analysis.append("Scalene triangle case not tested")
                elif 'Invalid' in line_content:
                    missing_analysis.append("Some invalid triangle cases not tested")
                elif 'return' in line_content:
                    missing_analysis.append(f"Return statement not covered: {line_content}")
        for branch in missing_branches:
            missing_analysis.append(f"Branch not covered at line {branch}")
        return "; ".join(missing_analysis) if missing_analysis else "General coverage needed"

    def _extract_test_code(self, generated_text: str) -> str:
        """Extract clean Python test code from Gemini's response"""
        generated_text = re.sub(r'```python\n?', '', generated_text)
        generated_text = re.sub(r'```\n?', '', generated_text)
        lines = generated_text.splitlines()
        test_lines = []
        in_parametrize = False
        open_brackets = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('@pytest.fixture') and not 'capture_output' in line:
                continue
            open_brackets += stripped.count('[') - stripped.count(']') + stripped.count('(') - stripped.count(')') + stripped.count('{') - stripped.count('}')
            if stripped.startswith('@pytest.mark.parametrize'):
                in_parametrize = True
                test_lines.append(line)
            elif in_parametrize and (open_brackets > 0 or stripped.startswith('(')):
                test_lines.append(line)
                if open_brackets == 0 and not stripped.endswith('['):
                    in_parametrize = False
            elif stripped.startswith(('import pytest', 'from triangle', '@pytest.fixture', 'def test_')):
                in_parametrize = False
                test_lines.append(line)
            elif test_lines and not in_parametrize and stripped:
                test_lines.append(line)
        result = '\n'.join(test_lines).rstrip()
        if open_brackets > 0:
            result += ']' * open_brackets
            print(f"✅ Fixed {open_brackets} unclosed brackets")
        return result

    def _validate_generated_code(self, code: str) -> bool:
        """Validate and attempt to fix syntax in generated code"""
        if not code.strip():
            print("⚠️ No code to validate")
            return False
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            print(f"⚠️ Syntax error: {e}")
            print(f"Problematic code:\n---\n{code[:500]}...\n---")
            if 'was never closed' in str(e):
                fixed_code = code + ']'
                try:
                    ast.parse(fixed_code)
                    with open('fixed_triangle_tests.py', 'w') as f:
                        f.write(fixed_code)
                    print("✅ Fixed syntax error by closing bracket")
                    return True
                except SyntaxError:
                    print("❌ Could not fix syntax error")
            return False

    def _generate_rule_based_tests(self, target_code: str, missing_lines: List[int], missing_branches: List, functions: List[Dict]) -> str:
        """Use Gemini to generate fallback tests instead of hard-coded cases"""
        print("🔄 Using Gemini for fallback test generation...")
        if self.model is None:
            print("❌ Gemini API unavailable, cannot generate fallback tests")
            return ""
        prompt = f"""Generate a complete pytest test file (`Triangle_test.py`) for the provided `triangle.py` to achieve 100% code and branch coverage, used as a fallback when primary generation fails.

**Target Code (`triangle.py`):**
```python
{target_code}
```

**Coverage Analysis:**
- Missing lines: {missing_lines}
- Missing branches: {missing_branches}
- Functions: {[f['name'] for f in functions]}

**Requirements:**
1. Include:
   - Imports: `pytest`, `triangle_type`, `is_valid_triangle`
   - Fixture: `capture_output` using `@pytest.fixture` and `capsys`
   - Tests for `is_valid_triangle` and `triangle_type`
2. Use `@pytest.mark.parametrize`
3. Cover all branches:
   - `is_valid_triangle`: `a > 0`, `b > 0`, `c > 0`, `a + b > c`, `b + c > a`, `c + a > b`
   - `triangle_type`: "Invalid", "Equilateral", "Isosceles", "Scalene"
4. Include edge cases:
   - Large numbers (e.g., 1e10)
   - Small numbers (e.g., 1e-6)
   - Floating-point (e.g., 2.5)
   - Boundary cases (e.g., a + b = c)
   - Zero/negative sides
5. Ensure syntax correctness:
   - Close all brackets ([], (), {{}})
   - Use `@pytest.fixture` only for `capture_output`
   - No invalid decorators
6. Accurate test cases:
   - `is_valid_triangle`: False for `(0.1, 0.1, 0.2)`, `(100, 1, 101)`
   - `triangle_type`: "Invalid" for invalid triangles
7. Return ONLY Python code
8. Use format:
```python
@pytest.mark.parametrize("a,b,c,expected", [(x,y,z,"result"), ...])
def test_name(a, b, c, expected, capture_output):
    actual = function_name(a, b, c)
    capture_output(actual)
    assert actual == expected
```

Generate the complete `Triangle_test.py`:
```python
import pytest
from triangle import triangle_type, is_valid_triangle

@pytest.fixture
def capture_output(capsys):
    def _capture(actual):
        print(f"Actual Output: {{actual}}")
        captured = capsys.readouterr()
        return actual
    return _capture
```"""
        try:
            generation_config = {
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2000,
            }
            response = self.model.generate_content(prompt, generation_config=generation_config)
            if response.text:
                generated_tests = self._extract_test_code(response.text)
                if self._validate_generated_code(generated_tests):
                    print("✅ Fallback test cases generated successfully")
                    return generated_tests
                else:
                    print("❌ Fallback code validation failed")
                    return ""
            else:
                print("❌ Empty fallback response from Gemini")
                return ""
        except Exception as e:
            print(f"❌ Error in fallback generation: {e}")
            return ""

class TestRunner:
    """Runs tests and reports results with actual and expected outputs"""

    def __init__(self, test_file: str):
        self.test_file = test_file

    def run_tests(self) -> List[Dict]:
        """Run all tests and return results with actual and expected outputs"""
        result = subprocess.run([
            sys.executable, '-m', 'pytest', self.test_file, '-v', '--tb=long', '--capture=tee-sys'
        ], capture_output=True, text=True, check=False)
        expected_outputs = self._parse_test_file_for_expected()
        test_results = self._parse_test_results(result.stdout, result.stderr, expected_outputs)
        return test_results

    def _parse_test_file_for_expected(self) -> Dict[str, List]:
        """Parse the test file to extract expected outputs from parametrized tests"""
        expected_outputs = {}
        try:
            with open(self.test_file, 'r') as f:
                content = f.read()
                tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_name = node.name
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and \
                           isinstance(decorator.func, ast.Attribute) and \
                           decorator.func.attr == 'parametrize':
                            param_names = decorator.args[0].value.split(',')
                            param_values = []
                            for arg in decorator.args[1].elts:
                                if isinstance(arg, (ast.List, ast.Tuple)):
                                    values = [ast.literal_eval(el) for el in arg.elts]
                                    param_values.append(values)
                            expected_outputs[test_name] = [
                                {'params': dict(zip(param_names, values)), 'expected': values[-1]}
                                for values in param_values
                            ]
                    if test_name not in expected_outputs:
                        expected = None
                        for stmt in node.body:
                            if isinstance(stmt, ast.Assert):
                                if isinstance(stmt.test, ast.Compare) and len(stmt.test.comparators) == 1:
                                    try:
                                        expected = ast.literal_eval(stmt.test.comparators[0])
                                    except Exception:
                                        continue
                        if expected is not None:
                            expected_outputs[test_name] = [{'params': {}, 'expected': expected}]
        except Exception as e:
            print(f"⚠️ Error parsing test file for expected outputs: {e}")
        return expected_outputs

    def _parse_test_results(self, stdout: str, stderr: str, expected_outputs: Dict[str, List]) -> List[Dict]:
        """Parse pytest output to extract test results, actual, and expected outputs"""
        results = []
        lines = stdout.splitlines()
        current_test = None
        in_failure = False
        actual_output = None
        for line in lines:
            if '::' in line and ('PASSED' in line or 'FAILED' in line or 'ERROR' in line):
                test_name_full = line.split('::')[-1].split()[0]
                test_name = test_name_full.split('[')[0] if '[' in test_name_full else test_name_full
                param_str = test_name_full[len(test_name)+1:-1] if '[' in test_name_full else None
                status = 'PASSED' if 'PASSED' in line else 'FAILED' if 'FAILED' in line else 'ERROR'
                expected = None
                if test_name in expected_outputs:
                    if param_str:
                        params = param_str.split('-')
                        for test_case in expected_outputs[test_name]:
                            param_values = [str(v) for v in test_case['params'].values()]
                            if all(str(p) in str(v) for p, v in zip(params, param_values[:-1])):
                                expected = test_case['expected']
                    else:
                        expected = expected_outputs[test_name][0].get('expected') if expected_outputs[test_name] else None
                current_test = {
                    'name': test_name_full,
                    'status': status,
                    'emoji': '✅' if status == 'PASSED' else '❌' if status == 'FAILED' else '⚠️',
                    'actual': None,
                    'expected': expected
                }
                results.append(current_test)
                in_failure = False
                actual_output = None
            elif 'FAILED' in line or 'ERROR' in line:
                in_failure = True
            elif in_failure and line.strip().startswith('E') and 'assert' in line:
                match = re.search(r"assert\s+(.+?)\s*==\s*(.+)", line)
                if match:
                    actual_output = match.group(1).strip("'\"")
                    try:
                        actual_output = ast.literal_eval(actual_output)
                    except Exception:
                        pass
                    if current_test:
                        current_test['actual'] = actual_output
            elif line.strip().startswith('Actual Output:'):
                actual_output = line.strip().split('Actual Output: ')[1].strip()
                try:
                    actual_output = ast.literal_eval(actual_output)
                except Exception:
                    pass
                if current_test:
                    current_test['actual'] = actual_output
        for result in results:
            if result['status'] == 'PASSED' and result['actual'] is None and result['expected'] is not None:
                result['actual'] = result['expected']
        return results

def main():
    """Main function to run the automatic testing process"""
    print("🚀 Starting Agentic Automatic Testing Tool")
    print(f"Target Module: {TARGET_MODULE}")
    print(f"Test File: {TEST_FILE}")
    print("-" * 50)

    if not os.path.exists(TARGET_MODULE):
        print(f"❌ Error: {TARGET_MODULE} not found!")
        return

    print(f"📝 Initializing {TEST_FILE}...")
    with open(TEST_FILE, 'w') as f:
        f.write("import pytest\nfrom triangle import triangle_type, is_valid_triangle\n")

    print("📊 Running initial coverage analysis...")
    coverage_analyzer = CoverageAnalyzer(TARGET_MODULE, TEST_FILE)
    initial_coverage = coverage_analyzer.run_coverage_analysis()
    print(f"Initial Statement Coverage: {initial_coverage['coverage_percentage']:.2f}%")
    print(f"Initial Branch Coverage: {initial_coverage['branch_coverage_percentage']:.2f}%")
    print(f"Missing lines: {initial_coverage['missing_lines']}")
    print(f"Missing branches: {initial_coverage['missing_branches']}")

    print("🤖 Generating test file with Gemini...")
    with open(TARGET_MODULE, 'r') as f:
        target_code = f.read()
    with open(TEST_FILE, 'r') as f:
        test_content = existing_tests = f.read()
    code_analyzer = CodeAnalyzer(TARGET_MODULE)
    functions = code_analyzer.extract_functions()
    gemini = GeminiTestGenerator(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
    new_tests = gemini.generate_test_cases(
        target_code,
        existing_tests,
        initial_coverage['missing_lines'],
        initial_coverage['missing_branches'],
        functions
    )
    if new_tests.strip():
        print(f"📝 Writing generated tests to {TEST_FILE}...")
        with open(TEST_FILE, 'w') as f:
            f.write(new_tests)
        print("📊 Running coverage analysis...")
        final_coverage = coverage_analyzer.run_coverage_analysis()
        print(f"Final Statement Coverage: {final_coverage['coverage_percentage']:.2f}%")
        print(f"Final Coverage: {final_coverage['branch_coverage_percentage']:.2f}%")
        print(f"Missing lines: {final_coverage['missing_lines']}")
        print(f"Missing branches: {final_coverage['missing_branches']}")
    else:
        print("❌ No valid tests generated by Gemini!")

    print("🧪 Running final tests...")
    test_runner = TestRunner(TEST_FILE)
    test_results = test_runner.run_tests()

    print("\n" + "="*50)
    print("📋 TEST RESULTS")
    print("="*=50)
    if not test_results:
        print("⚠️ No test results. Running tests manually...")
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', TEST_FILE, '-v', '--tb=long'],
            capture_output=True, text=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    else:
        for test in test_results:
            output = f"{test['emoji']} {test['name']} - {test['status']}"
            if test.get('actual') is not None and test.get('expected') is not None:
                output += f" (Actual: {test['actual']}, Expected: {test['expected']})"
            print(output)
    print("=" * 50)
    print("🎉 Testing completed!")

if __name__ == "__main__":
    try:
        import coverage
        import pytest
    except ImportError as e:
        print(f"Error: {e}")
        print("🚨🚕 Install missing packages:")
        print("pip install coverage pytest google-generativeai")
        import sys
        sys.exit(1)
    if GEMINI_API_KEY == "YOUR_API_KEY" or not GEMINI_API_KEY:
        print("🚨🚒 Set your API key in .env file")
        print("Get key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    main()
