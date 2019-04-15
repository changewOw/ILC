import re

regex = r".*"

a = bool(re.fullmatch(regex, "bn_conv1"))
print(a)