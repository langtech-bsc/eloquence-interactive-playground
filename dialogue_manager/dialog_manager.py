import json
import os
from typing import Dict, List, Optional, TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from gliner import GLiNER
from settings import settings

# Entity Ontology with description per entity (30 entity types)
ENTITIES = {
    "ABNORMALITY": 'Open-ended questions about anything unusual or changed (e.g., "anything unusual?", "have you noticed any changes?").',
    "AGE": 'Child\'s age or questions about age (e.g., "two months old", "how old is she?").',
    "BEHAVIOR": 'Observable actions or behaviour patterns (e.g., "crying", "refusing to eat", "sleepy", "asking for something").',
    "BODY_PART": 'Specific body part (e.g., "eye", "stomach", "diaper area", "ear", "throat").',
    "CAREGIVER_SUPPORT": 'Mentions of who helps the caregiver or household assistance (e.g., "grandma helps", "my husband watches the other kids", "no one else is home").',
    "CHARACTERISTIC": 'Adjectival qualities of signs/findings (color, texture, etc.) (e.g., "yellowish", "thick", "reddened", "scaly", "cold to touch").',
    "CHILD_TRAIT": 'Child\'s usual traits, appetite or temperament (e.g., "has a good appetite", "active", "easy-going", "loves to nurse").',
    "DEGREE": 'Intensity or severity modifiers (e.g., "mild", "moderate", "very high", "a bit", "slightly").',
    "DURATION": 'Time interval indicating how long something has lasted (e.g., "for two days", "over the past week", "last three hours", "for about ten minutes").',
    "EVENT": 'Any preceding event or possible cause (e.g., "fell off the bed", "choked on a toy", "hit his head", "had a seizure last night").',
    "ENVIRONMENT": 'Physical or situational context (e.g., "dark room", "in the park", "room temperature", "near the window").',
    "FEEDING": 'Feeding method or feeding events (method, schedule, refusal) (e.g., "eating", "breastfeeding", "bottle feeding every three hours", "refusing to nurse").',
    "FOOD": 'Specific foods or formulas (e.g., "apple", "porridge", "cereal", "formula").',
    "FREQUENCY": 'Repetition or rate (e.g., "every day", "sometimes", "twice a week", "occasionally", "three times a day").',
    "HYDRATION": 'Fluid intake or hydration-related remarks (e.g., "lots of water", "not drinking anything", "reduced urine output").',
    "INTERVENTION": 'Caregiver-initiated actions (e.g., "aspirated manually", "warmed with a blanket", "shook the bottle", "lifting the baby").',
    "KINDERGARTEN": 'Daycare/preschool status or recent changes (e.g., "in kindergarten", "not yet enrolled", "started daycare last week").',
    "MEDICAL_DEVICE": 'Tools or devices mentioned (e.g., "digital thermometer", "nasal aspirator", "pacifier", "home pulse oximeter", "bottle nipple").',
    "MOTHER_NUTRITION": 'Mother\'s dietary intake or eating/drinking remarks (e.g., "ate fruit", "drank only coffee all day", "not been eating", "taking prenatal vitamins").',
    "OUTCOME": 'Result or reaction after an action (e.g., "slept afterwards", "burped", "now she\'s fine", "fever went down").',
    "QUANTITY": 'Numeric amounts, dosages, counts or explicit numbers (e.g., "38.3°C", "two doses", "5 mL", "one sip", "three dirty diapers", "how many?").',
    "STOOL": 'Bowel movements or stool characteristics/frequency (only stool-related mentions) (e.g., "runny stool", "diarrheal stools", "three bowel movements today").',
    "SYMPTOM": 'Signs, sensations or bodily complaints (e.g., "fever", "pain", "mucus", "discharge", "wet diaper", "rash", "cough").',
    "TEST": 'Any measurement/checking action. (e.g., "measure the temperature").',
    "TIME": 'Specific points or references in time (e.g., "yesterday", "this morning", "3 PM", "a week ago").',
    "TREATMENT": 'Medications or remedies given (e.g., "Tylenol", "acetaminophen", "saline solution", "ibuprofen", "antibiotic").',
    "UNCERTAINTY": 'Expressions of doubt or approximation (e.g., "I think", "we\'re not sure", "maybe", "probably").',
    "URINATION": 'Urination events or wet-diaper status (e.g., "wet diaper", "no urine for 8 hours", "urinating normally", "only one wet diaper today").',
    "VACCINATION_STATUS": 'Immunization status or related remarks (e.g., "up-to-date", "delayed", "missed shots", "received vaccines last month").',
}

ENTITY_TYPES = [
    "ABNORMALITY",
    "AGE",
    "BEHAVIOR",
    "BODY_PART",
    "CAREGIVER_SUPPORT",
    "CHARACTERISTIC",
    "CHILD_TRAIT",
    "DEGREE",
    "DURATION",
    "EVENT",
    "ENVIRONMENT",
    "FEEDING",
    "FOOD",
    "FREQUENCY",
    "HYDRATION",
    "INTERVENTION",
    "KINDERGARTEN",
    "MEDICAL_DEVICE",
    "MOTHER_NUTRITION",
    "OUTCOME",
    "QUANTITY",
    "STOOL",
    "SYMPTOM",
    "TEST",
    "TIME",
    "TREATMENT",
    "UNCERTAINTY",
    "URINATION",
    "VACCINATION_STATUS",
]

# Entity priority list to ask the user
CATEGORY_PRIORITIES = {
    "fever": ["QUANTITY", "DURATION", "SYMPTOM", "BEHAVIOR", "INTERVENTION", "HYDRATION"],
    "cough": ["DURATION", "CHARACTERISTIC", "SYMPTOM", "EVENT", "FREQUENCY", "BEHAVIOR"],
    "rash": ["BODY_PART", "CHARACTERISTIC", "DURATION", "SYMPTOM", "FOOD", "ENVIRONMENT", "INTERVENTION"],
    "diarrhea": ["DURATION", "FREQUENCY", "STOOL", "CHARACTERISTIC", "HYDRATION", "BEHAVIOR", "ENVIRONMENT"],
    "constipation": ["DURATION", "STOOL", "FREQUENCY", "FEEDING", "HYDRATION", "INTERVENTION"],
    "injury": ["EVENT", "BODY_PART", "SYMPTOM", "OUTCOME", "BEHAVIOR", "INTERVENTION"],
    "feeding": ["FEEDING", "BEHAVIOR", "QUANTITY", "TIME", "FREQUENCY", "INTERVENTION"],
    "sleep": ["BEHAVIOR", "DURATION", "FREQUENCY", "ENVIRONMENT", "INTERVENTION", "OUTCOME"],
    "eye": ["BODY_PART", "CHARACTERISTIC", "DURATION", "SYMPTOM", "INTERVENTION", "EVENT"],
    "behavior": ["AGE", "TIME", "DURATION", "BEHAVIOR", "EVENT", "ENVIRONMENT", "INTERVENTION"],
    "development": ["AGE", "BEHAVIOR", "INTERVENTION", "CHILD_TRAIT", "ENVIRONMENT"],
    "post_vaccination": ["AGE", "TIME", "TREATMENT", "TEST", "QUANTITY", "DURATION", "BEHAVIOR", "FEEDING"],
    "cold": ["DURATION", "SYMPTOM", "TIME", "ENVIRONMENT", "FEEDING", "INTERVENTION", "BEHAVIOR"],
    "allergy": ["AGE", "TIME", "FOOD", "SYMPTOM", "BODY_PART", "STOOL", "CHARACTERISTIC"],
    "other": ["DURATION", "SYMPTOM", "BEHAVIOR", "FREQUENCY", "INTERVENTION", "ENVIRONMENT"],
}

CATEGORY_LIST = list(CATEGORY_PRIORITIES.keys())


# State schema
class DialogState(TypedDict):
    """Represents the current state of the dialog."""

    turn_count: int
    entities: Dict[str, List[str]]
    history: List[Dict[str, str]]
    category: Optional[str]
    last_query: Optional[str]
    terminated: bool
    last_user_utterance: Optional[str]
    ood: bool


class DialogManager:
    def __init__(self):
        """initialize dialog manager"""
        self.sessions = {}
        self._load_sessions()

        # Setup LLM backend -> TODO CHANGE it with BSC endpoint
        OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11435")
        OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")

        self.llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_predict=2048)

        # Load NER model -> TODO CHANGE with the BSC located finetuned GLiNER
        self.ner_model = GLiNER.from_pretrained(
            "gliner2_finetuned_uns",
            load_tokenizer=True,
        )
        self.ner_model.eval()

        # Initialize prompts
        self._init_prompts()

        # Build workflow
        self.workflow = self._build_workflow()

    def _init_prompts(self):
        """initialize prompt templates."""
        ### OOD Detection Prompt
        self.ood_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an Out-Of-Domain classifier for a pediatric medical call center service.

Last nurse question:
{last_system_utterance}

Given the user's utterance, determine if it is:
- on-topic: the user is describing the child's symptoms, history, or answering a question related to the child's health issue.
- off-topic: the user is asking for medical advice ("what should I do?", "is it serious?", "should I give medicine?"), or making irrelevant comments (e.g. asking for the weather).

If off-topic, output "OFF". If on-topic, output "ON". Return only the word.
""",
                ),
                ("user", "{utterance}"),
            ]
        )

        ### Category Detection Prompt
        self.category_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
You are a medical call center health issue complaint classifier.

Task:
Classify the parent's complaint about the child's health issue into EXACTLY ONE of the following categories:

{', '.join(CATEGORY_LIST)}

Category definitions:
- fever: elevated body temperature or feeling hot
- cough: dry or wet cough without primary cold symptoms
- rash: skin redness, bumps, itching, or irritation
- diarrhea: frequent loose or watery stools
- constipation: difficulty passing stool or infrequent bowel movements
- injury: physical trauma, fall, wound, burn, swelling from impact
- feeding: eating, breastfeeding, appetite, vomiting after feeding
- sleep: sleep problems, waking frequently, difficulty falling asleep
- eye: eye redness, discharge, swelling, irritation
- behavior: unusual crying, irritability, mood or behavioral changes
- post_vaccination: symptoms clearly occurring after a recent vaccine
- cold: runny nose, congestion, mild cough, typical cold symptoms
- allergy: sneezing, itchy eyes, rash triggered by allergen exposure
- other: use only if none of the above apply

Disambiguation rules:
- If fever happens after vaccination → post_vaccination
- If runny nose + mild cough → cold
- If itching/sneezing triggered by environment → allergy
- If symptom does not clearly match any category → other

Output rules:
- Return ONLY the category name.
- Do not explain.
- Do not return multiple categories.
- The output must be lowercase.
""",
                ),
                ("user", "{complaint}"),
            ]
        )

        ### Judge Prompt
        self.judge_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are the decision-maker for a pediatric medical call center dialog flow.
Your goal is to gather enough information so that a doctor can assess the situation.
You have a maximum of 10 questions. So far {turn_count} questions have been asked.

Below is the conversation history (system/nurse questions and parent/caller responses) and the entities already collected.
Based on this, decide the next action. You can either:
1. Ask a **new question** about a missing piece of information that is important for the case.
   - Do NOT repeat a question that has already been asked.
   - Do NOT give medical advice; only ask for facts.
   - The question should be concise and natural.
   - Your question should take into consideration the dialog history 
   - Your question should STRONGLY prioritize asking about missing entity type for the detected complaint category. 
   - If all priority entity types are collected, you can ask about any other relevant information or end.
2. If you believe you have collected enough information (all critical details are known, or the parent/caller has no more to add), output "END".

Consider STRONGLY the entity types when deciding what is still missing. For example, if the main symptom is fever, you need to know the temperature, duration, etc. If the child has a rash, you need to know its location, appearance, etc.

Your output must be a JSON object with exactly one field:
- If you want to ask a question: {{"action": "ask", "question": "your question here"}}
- If you want to end: {{"action": "end"}}

Conversation history:
{history}

Current entities:
{entities_json}

Missing priority entity types (ask about one of these):
{missing_entities}

Output JSON:""",
                )
            ]
        )

        ### Summary Prompt
        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a dialogue summarizer. Given the full conversation between a parent and a nurse, write a concise summary for the doctor.
Include: child's age, main complaint, key symptoms, duration, any relevant history, and any other important details. 
You are also given the entities referred at the dialogue and you need to include all entities referred into the dialog summary.
The summary should be in plain English, about 3-5 sentences, and ready to be read by a physician.""",
                ),
                (
                    "user",
                    "Entities:\n{entities}\n\nConversation:\n{history}\n\nDialog Summary:",
                ),
            ]
        )

    def _build_workflow(self):
        """Build the LangGraph workflow by defining the seuqence of states (nodes) and transitions (edges)"""
        workflow = StateGraph(DialogState)

        workflow.add_node("get_input", self._get_user_input)
        workflow.add_node("ood_detection", self._ood_detection)
        workflow.add_node("update_entities", self._update_entities)
        workflow.add_node("ask_question", self._ask_question)
        workflow.add_node("end", self._end_conversation)

        workflow.set_entry_point("get_input")
        workflow.add_edge("get_input", "ood_detection")
        workflow.add_conditional_edges(
            "ood_detection", self._after_ood, {"update": "update_entities", "reask": "get_input"}
        )
        workflow.add_edge("update_entities", "ask_question")
        workflow.add_conditional_edges("ask_question", self._should_continue, {"ask": "get_input", "end": "end"})
        workflow.add_edge("end", END)

        return workflow.compile()

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions

    def create_session(self, session_id: str, initial_state: DialogState):
        """Create a new session."""
        self.sessions[session_id] = initial_state.copy()
        self._save_sessions()

    def get_session_state(self, session_id: str) -> Optional[DialogState]:
        """Get session state."""
        return self.sessions.get(session_id)

    def add_to_history(self, session_id: str, role: str, text: str):
        """Add a message to session history."""
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({"role": role, "text": text})

    def update_last_query(self, session_id: str, query: str):
        """Update last query for session."""
        if session_id in self.sessions:
            self.sessions[session_id]["last_query"] = query

    def process_input(
        self, session_id: str, user_input: str, dialog_history: List[Dict[str, str]]
    ) -> tuple[str, DialogState]:
        """Process user input and return response and updated state."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        state = self.sessions[session_id].copy()
        state["last_user_utterance"] = user_input

        # Run the workflow
        result_state = self.workflow.invoke(state)

        # Update stored session
        self.sessions[session_id] = result_state
        self._save_sessions()

        # Get the last response
        if result_state["history"] and result_state["history"][-1]["role"] == "NURSE":
            response = result_state["history"][-1]["text"]
        else:
            response = "I understand. Let me process that."

        return response, result_state

    def reset_session(self, session_id: str) -> bool:
        """Reset a session."""
        if session_id in self.sessions:
            initial_state = DialogState(
                turn_count=0,
                entities={},
                history=[],
                category=None,
                last_query=None,
                last_user_utterance=None,
                terminated=False,
                ood=False,
            )
            self.sessions[session_id] = initial_state
            self._save_sessions()
            return True
        return False

    def end_session(self, session_id: str) -> tuple[Optional[str], Optional[DialogState]]:
        """End a session and generate summary."""
        if session_id not in self.sessions:
            return None, None

        state = self.sessions[session_id]
        summary = self._generate_summary(state["history"], state["entities"])

        # Mark as terminated
        state["terminated"] = True
        self._save_sessions()

        return summary, state

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)

    # Dialog Manager Core Methods (from your code)

    def _is_ood(self, utterance: str, last_system_utterance: str) -> bool:
        """Return True if utterance is OOD / off-topic."""
        prompt = self.ood_prompt.format(utterance=utterance, last_system_utterance=last_system_utterance)
        response = self.llm.invoke(prompt)
        result = response.content.strip().upper()
        return result == "OFF"

    def _extract_entities(self, utterance: str) -> Dict[str, List[str]]:
        """Extract entities from utterance using GLiNER."""
        available_entities = ENTITIES

        predicted_entities = self.ner_model.predict_entities(utterance, available_entities, threshold=0.5)

        if isinstance(predicted_entities, list):
            entity_dict: Dict[str, List[str]] = {}
            for entity in predicted_entities:
                label = entity.get("label")
                text = entity.get("text")
                if label in ENTITY_TYPES:
                    entity_dict.setdefault(label, []).append(text)
            return entity_dict
        return {}

    def _detect_category(self, initial_query: str) -> str:
        """Determine complaint category from first utterance."""
        prompt = self.category_prompt.format(complaint=initial_query)
        response = self.llm.invoke(prompt)
        cat = response.content.strip().lower()
        if cat not in CATEGORY_LIST:
            return "other"
        return cat

    def _judge_action(self, state: DialogState) -> Dict[str, str]:
        """LLM decides next action."""
        history_str = ""
        for turn in state["history"]:
            role = turn["role"].capitalize()
            text = turn["text"]
            history_str += f"{role}: {text}\n"
        entities_json = json.dumps(state["entities"], ensure_ascii=False)

        # Find missing entity types
        category = state.get("category", "other")
        priority_list = CATEGORY_PRIORITIES.get(category, CATEGORY_PRIORITIES["other"])
        missing = []
        for etype in priority_list:
            if etype not in state["entities"] or not state["entities"][etype]:
                missing.append(etype)
                break

        missing_str = ", ".join(missing) if missing else "None - you may ask any relevant question or end."

        prompt = self.judge_prompt.format(
            turn_count=state["turn_count"],
            history=history_str,
            entities_json=entities_json,
            missing_entities=missing_str,
        )
        parser = JsonOutputParser()
        chain = self.llm | parser
        try:
            result = chain.invoke(prompt)
            if "action" not in result:
                return {"action": "ask", "question": "Can you tell me more about that?"}
            return result
        except Exception as e:
            print(f"Judge error: {e}", flush=True)
            return {
                "action": "ask",
                "question": "Is there anything else you'd like to share?",
            }

    def _generate_summary(self, history: List[Dict[str, str]], entities: Dict[str, List[str]]) -> str:
        """Generate dialog summary."""
        history_str = ""
        for turn in history:
            role = turn["role"].upper()
            text = turn["text"]
            history_str += f"{role}: {text}\n"

        entities_json = json.dumps(entities, ensure_ascii=False, indent=2)
        prompt = self.summary_prompt.format(
            history=history_str,
            entities=entities_json,
        )
        response = self.llm.invoke(prompt)
        return response.content.strip()

    # LangGraph Node Methods
    def _get_user_input(self, state: DialogState) -> DialogState:
        """Process user input."""
        state["ood"] = False

        # Check if this is the first turn and no welcome message yet
        if not state.get("history") or len(state["history"]) == 0:
            welcome = "Hello, this is the pediatric medical call center. How can I help you?"
            state["history"].append({"role": "NURSE", "text": welcome})
            state["last_query"] = welcome

        # If there's no user input to process (first turn), return state
        if not state.get("last_user_utterance"):
            return state

        return state

    def _ood_detection(self, state: DialogState) -> DialogState:
        """Check if the latest user utterance is off-topic."""
        utterance = state.get("last_user_utterance", "")
        if not utterance or state.get("terminated", False):
            return state

        last_system_utterance = state.get(
            "last_query",
            "Hello, this is the pediatric medical call center. How can I help you?",
        )

        if self._is_ood(utterance, last_system_utterance):
            # OOD detected: respond with standard message and re-ask last question
            ood_msg = (
                "I am sorry but I can only help gather information for the doctor regarding your child's situation."
            )
            state["history"].append({"role": "NURSE", "text": ood_msg})

            # Re-ask the last query
            reask = state.get(
                "last_query",
                "Hello, this is the pediatric medical call center. How can I help you?",
            )
            state["history"].append({"role": "NURSE", "text": reask})
            state["ood"] = True
            state.pop("last_user_utterance", None)

            return state

        return state

    def _update_entities(self, state: DialogState) -> DialogState:
        """Extract entities from latest parent/caller utterance."""
        utterance = state.pop("last_user_utterance", "")
        if not utterance or state.get("terminated", False):
            return state

        state["history"].append({"role": "CALLER", "text": utterance})

        # Detect category on first parent utterance
        if len([t for t in state["history"] if t["role"] == "CALLER"]) == 1:
            state["category"] = self._detect_category(utterance)

        # Extract entities
        new_entities = self._extract_entities(utterance)
        if new_entities:
            for etype, values in new_entities.items():
                if etype not in state["entities"]:
                    state["entities"][etype] = []
                for val in values:
                    if val not in state["entities"][etype]:
                        state["entities"][etype].append(val)

        return state

    def _ask_question(self, state: DialogState) -> DialogState:
        """Use LLM judge to decide next question."""
        decision = self._judge_action(state)

        # End if turn limit reached or judge decides to end
        if state["turn_count"] >= 10 or decision["action"] == "end":
            state["terminated"] = True
            return state

        question = decision.get("question", "Could you tell me more?")
        state["last_query"] = question
        state["history"].append({"role": "NURSE", "text": question})
        state["turn_count"] += 1

        return state

    def _end_conversation(self, state: DialogState) -> DialogState:
        """Generate summary and mark terminated."""
        closing_msg = "Thank you. I am connecting you with a doctor now. Please stay on the line."
        state["history"].append({"role": "NURSE", "text": closing_msg})
        state["terminated"] = True
        return state

    def _should_continue(self, state: DialogState) -> str:
        """Conditional edge: continue if not terminated and turn_count < 10."""
        if state.get("terminated", False) or state["turn_count"] >= 10:
            return "end"
        return "ask"

    def _after_ood(self, state: DialogState) -> str:
        """After OOD, decide next step."""
        if state.get("ood", False):
            return "reask"
        return "update"

    def _load_sessions(self):
        """Load sessions from disk."""
        if os.path.exists(settings.DIALOG_SESSIONS_PATH):
            try:
                with open(settings.DIALOG_SESSIONS_PATH, "r") as f:
                    self.sessions = json.load(f)
            except:
                self.sessions = {}

    def _save_sessions(self):
        """Save sessions to disk."""
        os.makedirs(os.path.dirname(settings.DIALOG_SESSIONS_PATH), exist_ok=True)
        with open(settings.DIALOG_SESSIONS_PATH, "w") as f:
            json.dump(self.sessions, f, indent=2)
