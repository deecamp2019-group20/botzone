# Botzone脚本

先把预训练模型文件放到自己的存储空间里，提交的bot脚本是script.py。
可能需要删除开头一些不必要的import。

使用payload方式传环境观测数据的，实现 choose_action_by_payload 方法，返回一个长度为15的list，每个元素表示3/4/5/.../14/15出多少张，全零表示不要（要不起）。

