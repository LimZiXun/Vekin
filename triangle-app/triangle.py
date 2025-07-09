
def is_valid_triangle(a, b, c):
    return a > 0 and b > 0 and c > 0 and (a + b > c) and (b + c > a) and (a + c > b)

def triangle_type(a, b, c):
    if not is_valid_triangle(a, b, c):
        return "Invalid"
    if a == b == c:
        return "Equilateral"
    if a == b or b == c or a == c:
        return "Isosceles"
    return "Scalene"