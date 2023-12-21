def hello():
    print("Hello, World!")

# 将函数名存储在字符串中
function_name = "hello"

# 使用 globals() 或 locals() 函数获取函数对象
function = globals().get(function_name)

# 检查函数是否存在
if function is not None and callable(function):
    # 调用函数
    function()
else:
    print(f"Function '{function_name}' does not exist.")