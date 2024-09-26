import json

# 用JSON格式模拟数据库j
class CourseDatabase:
    def __init__(self):
        self.database = {
            "大模型技术实战":{
                "课时": 200,
                "每周更新次数": 3,
                "每次更新小时": 2
            },
             "机器学习实战":{
                "课时": 230,
                "每周更新次数": 2,
                "每次更新小时": 1.5
            },
            "深度学习实战":{
                "课时": 150,
                "每周更新次数": 1,
                "每次更新小时": 3
            },
            "AI数据分析":{
                "课时": 10,
                "每周更新次数": 1,
                "每次更新小时": 1
            },
        }
    def course_query(self, course_name):
        return self.database.get(course_name, "目前没有该课程信息")


# 定义数据库操作工具
class CourseOperations:
    def __init__(self):
        self.db = CourseDatabase()

    def add_hours_to_course(self, course_name, additional_hours):
        if course_name in self.db.database:
            self.db.database[course_name]['课时'] += int(additional_hours)
            return f"课程 {course_name}的课时已增加{additional_hours}小时。"
        else:
            return "课程不存在,无法添加课时"
