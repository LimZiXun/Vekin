import pytest
from triangle import triangle_type, is_valid_triangle

@pytest.fixture
def capture_output(capsys):
    """Fixture to capture actual output for each test"""
    def _capture(actual):
        print(f"Actual Output: {actual}")
        captured = capsys.readouterr()
        return actual
    return _capture

@pytest.mark.parametrize("a,b,c,expected", [
    (3, 4, 5, True),
    (5, 5, 5, True),
    (1, 2, 3, False),
    (0, 1, 1, False),
    (-1, 2, 2, False),
])
def test_is_valid_triangle(a, b, c, expected, capture_output):
    """Test is_valid_triangle function"""
    actual = is_valid_triangle(a, b, c)
    capture_output(actual)
    assert actual == expected

@pytest.mark.parametrize("a,b,c,expected", [
    (3, 3, 3, "Equilateral"),
])
def test_equilateral_triangle(a, b, c, expected, capture_output):
    """Test equilateral triangle case"""
    actual = triangle_type(a, b, c)
    capture_output(actual)
    assert actual == expected

@pytest.mark.parametrize("a,b,c,expected", [
    (0, 0, 0, "Invalid"),
])
def test_invalid_triangle_additional(a, b, c, expected, capture_output):
    """Test additional invalid triangle cases"""
    actual = triangle_type(a, b, c)
    capture_output(actual)
    assert actual == expected

@pytest.mark.parametrize("a,b,c,expected", [
    (5, 3, 5, "Isosceles"),
])
def test_isosceles_triangle(a, b, c, expected, capture_output):
    """Test isosceles triangle combinations"""
    actual = triangle_type(a, b, c)
    capture_output(actual)
    assert actual == expected

@pytest.mark.parametrize("a,b,c,expected", [
    (4, 5, 6, "Scalene"),
])
def test_scalene_triangle(a, b, c, expected, capture_output):
    """Test scalene triangle case"""
    actual = triangle_type(a, b, c)
    capture_output(actual)
    assert actual == expected