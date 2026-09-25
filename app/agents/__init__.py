from app.agents.supervisor import supervisor_agent, SupervisorAgent
from app.agents.web_agent import web_agent, WebAgent
from app.agents.rag_agent import rag_agent, RAGAgent
from app.agents.code_agent import code_agent, CodeAgent
from app.agents.general_agent import general_agent, GeneralAgent
from app.agents.critic_agent import critic_agent, CriticAgent
from app.agents.final_agent import final_agent, FinalAgent

__all__ = [
    "supervisor_agent",
    "SupervisorAgent",
    "web_agent",
    "WebAgent",
    "rag_agent",
    "RAGAgent",
    "code_agent",
    "CodeAgent",
    "general_agent",
    "GeneralAgent",
    "critic_agent",
    "CriticAgent",
    "final_agent",
    "FinalAgent",
]
