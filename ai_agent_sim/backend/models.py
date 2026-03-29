from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import enum

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    openId = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(Text, nullable=True)
    email = Column(String(320), nullable=True)
    loginMethod = Column(String(64), nullable=True)
    role = Column(SQLEnum("user", "admin"), default="user", nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    lastSignedIn = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    simulations = relationship("Simulation", back_populates="user")

class Simulation(Base):
    __tablename__ = "simulations"
    
    id = Column(Integer, primary_key=True, index=True)
    userId = Column(Integer, ForeignKey("users.id"), nullable=False)
    postText = Column(Text, nullable=False)
    predictedScore = Column(Float, nullable=False)
    agentsSpawned = Column(Integer, nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="simulations")
    responses = relationship("AgentResponse", back_populates="simulation")

class AgentResponse(Base):
    __tablename__ = "agent_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    simulationId = Column(Integer, ForeignKey("simulations.id"), nullable=False)
    agentName = Column(String(128), nullable=False)
    agentPersona = Column(String(128), nullable=False)
    response = Column(Text, nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    simulation = relationship("Simulation", back_populates="responses")
