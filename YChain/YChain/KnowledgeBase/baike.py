# 用于将百科信息处理为知识库
# from text2vec_handle import *

# baike_dict_list =[
#     {'林黛玉':'''林黛玉，中国古典名著《红楼梦》的女主角，金陵十二钗正册双首之一，西方灵河岸绛珠仙草转世，荣府幺女贾敏与扬州巡盐御史林如海之独生女，母亲贾敏是贾代善和贾母四个女儿里最小的女儿，贾母的外孙女，贾宝玉的姑表妹、恋人、知己，贾府通称林姑娘。林黛玉是古代文学作品中极富灵气的经典女性形象。从小聪明清秀，父母对她爱如珍宝。5岁上学，6至7岁母亲早亡，10岁接到贾母身边抚养教育。11岁时父亲逝世，从此常住贾府，养成了孤标傲世的性格。''',},
#     {'倒拔垂杨柳':'''倒拔垂杨柳是古典名著《水浒传》的故事情节。《花和尚倒拔垂杨柳》选自于《水浒传》第七回《花和尚倒拔垂杨柳 豹子头误入白虎堂》。鲁智深为了收服一众泼皮，用左手向下搂住树干，右手把住树的上半截，腰往上一挺，竟将杨柳树连根拔起。''',},
#     {'055型驱逐舰':'''055型驱逐舰（英文：Type 055 destroyer，北约代号：Renhai-class ，译文：刃海级）是中国研制的新型舰队防空驱逐舰。中国首艘055型导弹驱逐舰2018年8月24日上午离开上海江南造船厂码头，进行首次海上测试。055型导弹驱逐舰生逢其时、肩负重任，它对于中国海军实施“近海防御、远海护卫”战略具有重要意义。在远海大洋作战，需要面对的是拥有全维空间作战能力的强大对手，以防空、反导、反潜、反舰等高端、高强度作战为主要内容。防空反导能力突出的055型导弹驱逐舰批量服役后，可为中国海军航母战斗群和水面舰艇战斗群撑起更加可靠的“空中保护伞”，为其在远海大洋遂行任务提供更好的保证。''',},
#     {'唐纳德·特朗普':'''唐纳德·特朗普（Donald Trump，1946年6月14日- ），出生于美国纽约，祖籍德国巴伐利亚自由州，德裔美国共和党籍政治家、企业家、房地产商人、电视人，第45任美国总统（2017年1月20日-2021年1月20日）。''',}
# ]



def get_title_list(baike_dict_list):
    return [list(item.keys())[0] for item in baike_dict_list]
def get_description_list(baike_dict_list):
    return [list(item.values())[0] for item in baike_dict_list]
def get_text_list(baike_dict_list):
    return [list(item.values())[0] for item in baike_dict_list]



# def baike_title_sim(query,baike_dict_list,text2vec_model):

#     query_embeddings = text_2_vec(text2vec_model,query,batch_size=1)
#     baike_embeddings = text_2_vec(text2vec_model,baike_dict_list,batch_size=1)
#     index,_ = find_most_similar_data(query_embeddings, baike_embeddings,text2vec_model)
#     baike_info = baike_dict_list[index].keys() +':'+ baike_dict_list[index].values()
#     return baike_info

# def baike_description_sim(query,baike_dict_list,text2vec_model):
    
#     return

# def baike_text_sim(query,baike_dict_list,text2vec_model):
    
#     return
