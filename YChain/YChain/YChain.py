# 这是我自己写的chain
from YChain.Text2Vec import *





def get_prompt(prompt_template,text,prompt):
    return prompt_template.replace('{text}',text).replace('{prompt}',prompt)

def chat(llm,prompt_template,text,prompt,history=[],show_input=True):
    
    prompt = get_prompt(prompt_template,text,prompt)
    if show_input == True:
        print()
        print('输入格式：')
        print(prompt)
        print()
    rst,history = llm.call(prompt,history)
    return rst,history

def chat_with_data(prompt,embedding_vectors,history,prompt_template,data_list,text2vec_model,llm,show_input=True):
    index,_ = find_most_similar_data(prompt, embedding_vectors,text2vec_model)
    data_text = data_list[index]
    print('==知识库检索成功')
    response,history = chat(llm,prompt_template,data_text,prompt,history=history,show_input=show_input)
    return index,response,history
