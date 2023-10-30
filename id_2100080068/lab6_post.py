def xor(x1, x2):
  net = x1 + x2
  threshold = 1
  if net >= threshold:
    return 1
  else:
    return 0
print(xor(0, 0))
print(xor(0, 1))
print(xor(1, 0))
print(xor(1, 1))
