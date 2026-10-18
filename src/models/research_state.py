"""
Data models for the research agent
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class QuestionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"




@dataclass
class SubQuestion:
    """Represents a research sub-question"""
    id: str
    question: str
    status: QuestionStatus = QuestionStatus.PENDING
    priority: int = 0  # Higher = more important
    depth: int = 0  # How many levels deep in the question tree
    answer: Optional[str] = None


@dataclass
class SearchResult:
    """Represents a single search result"""
    url: str
    title: str
    content: str
    score: float
    full_content: Optional[str] = None  # Full webpage content
    category: Optional[str] = None  # social, educational, etc.
    question_id: Optional[str] = None


@dataclass
class KnowledgeFinding:
    """A piece of knowledge extracted from research"""
    content: str
    sources: List[str]
    confidence: float  # 0-1
    question_id: str
    contradictions: List[str] = field(default_factory=list)


@dataclass
class ResearchGap:
    """Represents an identified gap in knowledge"""
    description: str
    question_id: str
    severity: str  # critical, moderate, minor
    suggested_searches: List[str] = field(default_factory=list)


@dataclass
class ReportSection:
    """A section of the final report"""
    title: str
    content: str
    sources: List[str]
    order: int


@dataclass
class ResearchState:
    """Tracks the overall state of the research process"""
    main_query: str
    category_toggles: Dict[str, bool]
    
    # Question management
    questions: List[SubQuestion] = field(default_factory=list)
    question_queue: List[str] = field(default_factory=list)  # IDs in priority order
    completed_questions: Set[str] = field(default_factory=set)
    
    # Search results and findings
    search_results: List[SearchResult] = field(default_factory=list)
    findings: List[KnowledgeFinding] = field(default_factory=list)
    gaps: List[ResearchGap] = field(default_factory=list)
    
    # Metadata
    iteration_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_searches: int = 0
    
    # Report
    report_sections: List[ReportSection] = field(default_factory=list)
    final_report: Optional[str] = None
    
    # Research reasoning steps for debugging/transparency
    reasoning_steps: List[str] = field(default_factory=list)
    
    def get_question_by_id(self, question_id: str) -> Optional[SubQuestion]:
        """Get a question by its ID"""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None
    
    def add_question(self, question: SubQuestion):
        """Add a new question to the state"""
        self.questions.append(question)
        if question.status == QuestionStatus.PENDING:
            self.question_queue.append(question.id)
    
    def mark_question_completed(self, question_id: str):
        """Mark a question as completed"""
        question = self.get_question_by_id(question_id)
        if question:
            question.status = QuestionStatus.COMPLETED
            self.completed_questions.add(question_id)
            if question_id in self.question_queue:
                self.question_queue.remove(question_id)
    
    def get_next_question(self) -> Optional[SubQuestion]:
        """Get the next question to process"""
        if not self.question_queue:
            return None
        
        # Sort by priority (descending)
        sorted_ids = sorted(
            self.question_queue,
            key=lambda qid: self.get_question_by_id(qid).priority if self.get_question_by_id(qid) else 0,
            reverse=True
        )
        
        if sorted_ids:
            next_id = sorted_ids[0]
            question = self.get_question_by_id(next_id)
            if question:
                question.status = QuestionStatus.IN_PROGRESS
            return question
        return None
    
    def get_findings_for_question(self, question_id: str) -> List[KnowledgeFinding]:
        """Get all findings related to a specific question"""
        return [f for f in self.findings if f.question_id == question_id]
