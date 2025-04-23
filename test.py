from docx import Document


def docx_to_text(docx_path, txt_path):
    # 加载docx文件
    doc = Document(docx_path)

    # 初始化一个空字符串来存储文本内容
    full_text = []

    # 遍历文档中的所有段落
    for para in doc.paragraphs:
        full_text.append(para.text)

    # 将文档内容连接成一个字符串（可选，取决于是否需要保留段落分隔）
    text = '\n'.join(full_text)

    # 写入到txt文件
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"转换完成，内容已保存到 {txt_path}")


# 使用函数
docx_path = 'lps/demo2.docx'  # 替换为你的.docx文件路径
txt_path = 'lps/demo2.txt'  # 替换为你希望输出的.txt文件路径
docx_to_text(docx_path, txt_path)