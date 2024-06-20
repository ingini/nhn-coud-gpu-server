# MySQL 모델 정의
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class QuestionAnswer(Base):
    __tablename__ = "question_answers"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String(255), index=True)
    answer = Column(Text)
    type = Column(String, nullable=True)
    link = Column(String, nullable=True)