from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True)
    title = Column(String)
    language = Column(String)
    creation_date = Column(String)
    content = Column(Text)
    embedding = Column(LargeBinary)

class SearchHistory(Base):
    __tablename__ = 'search_history'
    id = Column(Integer, primary_key=True)
    query = Column(String)
    document_id = Column(Integer, ForeignKey('documents.id'))
    timestamp = Column(DateTime)
    document = relationship("Document")

DATABASE_PATH = '/home/solomon/data/lose_data/database.db'
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
