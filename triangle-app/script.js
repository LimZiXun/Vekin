function checkTriangle() {
    const side1 = parseFloat(document.getElementById('side1').value);
    const side2 = parseFloat(document.getElementById('side2').value);
    const side3 = parseFloat(document.getElementById('side3').value);
    
    // Fixed Bug 1: Proper input validation
    if (isNaN(side1) || isNaN(side2) || isNaN(side3) || 
        side1 <= 0 || side2 <= 0 || side3 <= 0) {
        showResult('Please enter valid positive numbers for all sides');
        return;
    }

    // Fixed Bug 2: Proper triangle inequality theorem check
    // Sum of any two sides must be strictly greater than the third side
    if (side1 + side2 > side3 && side2 + side3 > side1 && side1 + side3 > side2) {
        // Bug 1: Wrong condition for equilateral triangle
        // Should check if all sides are equal, but checks if any two sides are equal
        if (side1 === side2 || side2 === side3 || side1 === side3) {
            showResult('This is an equilateral triangle');//side1 === side2 === side3
        } 
        // Bug 2: Wrong condition for scalene triangle
        // Should check if all sides are different, but checks if any two sides are different
        else if (side1 !== side2 || side2 !== side3 || side1 !== side3) {// side1 !== side2 !== side3
            showResult('This is a scalene triangle');
        } else {
            showResult('This is an isosceles triangle');
        }
    } else {
        showResult('These sides cannot form a triangle');
    }
}

function showResult(message) {
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = message;
    resultDiv.style.backgroundColor = '#e8f5e9';
    resultDiv.style.color = '#2e7d32';
} 