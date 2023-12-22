# 打开文件
file = open("output.txt", "w")  # 'w' 参数表示以写入模式打开文件

# 字符串内容
content = "这是要写入文件的字符串"

# 将字符串写入文件
file.write(content)

# 关闭文件
file.close()
