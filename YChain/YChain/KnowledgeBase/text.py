# 处理文本的一些功能 




# 用于将文本处理为知识库
# 自动划分长文本部分
def text_splitter(text,chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
# 滑动窗口
def text_splitter_sliding_window(text, window_size=1000, step_size=500):
    text_length = len(text)
    segments = []
    for i in range(0, text_length - window_size + 1, step_size):
        segment = text[i:i + window_size]
        segments.append(segment)
    return segments

