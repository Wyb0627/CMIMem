"""
Format Checker for Memory Agent Outputs.

Validates that policy model outputs follow the correct JSON format
for each memory agent type.

Includes truncation-tolerant parsing: when responses are cut off by
max_response_length, the checker extracts all complete operation entries
from the truncated JSON rather than failing outright.
"""

import json
import re
from typing import Dict, List, Optional, Tuple


class FormatChecker:
    """Format validation for memory agent outputs."""

    REQUIRED_AGENTS = ['core', 'episodic', 'semantic', 'procedural']

    VALID_CORE_OPS = ['APPEND', 'REPLACE', 'REWRITE']
    VALID_CORE_OPS_EXTRACTED = ['APPEND', 'REPLACE', 'REWRITE', 'SKIP']

    VALID_ACTIONS_BY_AGENT = {
        'episodic': ['ADD', 'UPDATE', 'MERGE'],
        'semantic': ['ADD', 'UPDATE', 'SKIP'],
        'procedural': ['ADD', 'UPDATE']
    }

    VALID_ACTIONS_BY_AGENT_EXTRACTED = {
        'episodic': ['ADD', 'UPDATE', 'MERGE', 'SKIP'],
        'semantic': ['ADD', 'UPDATE', 'SKIP'],
        'procedural': ['ADD', 'UPDATE', 'SKIP']
    }

    def __init__(self):
        pass

    def _clean_think_tags(self, response: str) -> str:
        """Remove think tags from response."""
        if not response:
            return response

        clean = response.strip()

        if '' in clean:
            clean = clean.split('</think>')[-1].strip()

        clean = re.sub(r'<thinking>.*?</thinking>', '', clean, flags=re.DOTALL)

        return clean.strip()

    def check_format(self, response: str, agent_type: str,
                     extracted_mode: bool = False) -> Tuple[bool, Optional[Dict]]:
        """
        Check if response has valid format for the given agent type.

        Args:
            response: Raw model output
            agent_type: One of 'core', 'episodic', 'semantic', 'procedural'
            extracted_mode: If True, uses extracted candidate mode. The model
                still outputs standard text format (content/memory fields),
                so we reuse original validators but with relaxed action sets
                (e.g. SKIP is allowed for all agents in extracted mode).

        Returns:
            Tuple of (is_valid, parsed_dict or None)
        """
        if not response:
            return False, None

        clean = self._clean_think_tags(response)

        # Remove markdown code blocks
        clean = re.sub(r'```(?:json)?\s*', '', clean)
        clean = re.sub(r'\s*```', '', clean)
        clean = clean.strip()

        # Extract JSON object
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not json_match:
            json_start = clean.find('{')
            if json_start >= 0:
                parsed = self._try_repair_truncated_json(clean[json_start:], agent_type)
                if parsed is not None:
                    if agent_type == 'core':
                        return self._validate_core(parsed, allow_skip=extracted_mode)
                    else:
                        return self._validate_memory_agent(parsed, agent_type, allow_skip=extracted_mode)
            return False, None

        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            parsed = self._try_repair_truncated_json(json_match.group(0), agent_type)
            if parsed is None:
                return False, None

        if agent_type == 'core':
            return self._validate_core(parsed, allow_skip=extracted_mode)
        else:
            return self._validate_memory_agent(parsed, agent_type, allow_skip=extracted_mode)

    def _try_repair_truncated_json(self, json_text: str, agent_type: str) -> Optional[Dict]:
        """Attempt to repair truncated JSON by extracting complete operations.

        When responses are truncated by max_response_length, the JSON is cut
        mid-way. This method finds the last complete operation entry and
        reconstructs a valid JSON object from complete entries only.
        """
        if agent_type == 'core':
            # Core has simple structure, try closing braces
            for suffix in ['"}', '"}}', '}']:
                try:
                    return json.loads(json_text + suffix)
                except json.JSONDecodeError:
                    continue
            return None

        # For memory agents (episodic/semantic/procedural), find complete operations
        # Strategy: find all complete {...} objects within the "operations" array
        ops_match = re.search(r'"operations"\s*:\s*\[', json_text)
        if not ops_match:
            return None

        array_start = ops_match.end()
        complete_ops = self._extract_complete_objects(json_text, array_start)

        if not complete_ops:
            return None

        return {"operations": complete_ops}

    def _extract_complete_objects(self, text: str, start_pos: int) -> List[Dict]:
        """Extract all complete JSON objects from a potentially truncated array."""
        complete_ops = []
        brace_depth = 0
        current_obj_start = None

        for i in range(start_pos, len(text)):
            char = text[i]
            if char == '{':
                if brace_depth == 0:
                    current_obj_start = i
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and current_obj_start is not None:
                    obj_str = text[current_obj_start:i + 1]
                    try:
                        obj = json.loads(obj_str)
                        complete_ops.append(obj)
                    except json.JSONDecodeError:
                        pass
                    current_obj_start = None

        return complete_ops

    def _validate_core(self, parsed: Dict, allow_skip: bool = False) -> Tuple[bool, Optional[Dict]]:
        """Validate core memory agent output.

        Args:
            parsed: Parsed JSON dict from model output.
            allow_skip: If True, also accept SKIP operation (used in extracted mode).
        """
        operation = parsed.get('operation', '').upper()

        valid_ops = self.VALID_CORE_OPS_EXTRACTED if allow_skip else self.VALID_CORE_OPS
        if operation not in valid_ops:
            return False, None

        if operation == 'SKIP':
            return True, parsed
        elif operation == 'APPEND':
            if 'content' not in parsed:
                return False, None
        elif operation == 'REPLACE':
            if 'old_text' not in parsed or 'new_text' not in parsed:
                return False, None
        elif operation == 'REWRITE':
            if 'content' not in parsed:
                return False, None

        return True, parsed

    def _validate_memory_agent(self, parsed: Dict, agent_type: str,
                               allow_skip: bool = False) -> Tuple[bool, Optional[Dict]]:
        """Validate episodic/semantic/procedural agent output.

        Args:
            parsed: Parsed JSON dict from model output.
            agent_type: One of 'episodic', 'semantic', 'procedural'.
            allow_skip: If True, use the extended action set that includes SKIP
                for all agent types (used in extracted mode).
        """
        operations = parsed.get('operations', [])

        if not isinstance(operations, list):
            return False, None

        valid_actions = (self.VALID_ACTIONS_BY_AGENT_EXTRACTED if allow_skip
                         else self.VALID_ACTIONS_BY_AGENT).get(agent_type, [])

        for op in operations:
            action = op.get('action', '').upper()
            if action not in valid_actions:
                return False, None

            # Validate required fields
            if action == 'ADD':
                if 'memory' not in op:
                    return False, None
            elif action == 'UPDATE':
                if 'old_memory' not in op or 'new_memory' not in op:
                    return False, None
            elif action == 'MERGE':
                if 'old_memories' not in op or 'new_memory' not in op:
                    return False, None
            elif action == 'SKIP':
                # In extracted mode, reason is optional (model may just skip)
                pass

        return True, parsed

    def _validate_core_extracted(self, parsed: Dict) -> Tuple[bool, Optional[Dict]]:
        """Validate core memory agent output in extracted mode.

        Expected formats:
          {"operation": "APPEND", "item_ids": [1, 3]}
          {"operation": "REPLACE", "old_text": "...", "item_ids": [2]}
          {"operation": "REWRITE", "item_ids": [1, 2, 5]}
          {"operation": "SKIP"}
        """
        operation = parsed.get('operation', '').upper()

        if operation not in self.VALID_CORE_OPS_EXTRACTED:
            return False, None

        if operation == 'SKIP':
            return True, parsed

        # APPEND / REPLACE / REWRITE all require item_ids
        item_ids = parsed.get('item_ids', [])
        if not isinstance(item_ids, list) or not item_ids:
            return False, None
        if not all(isinstance(x, int) for x in item_ids):
            return False, None

        if operation == 'REPLACE':
            if 'old_text' not in parsed:
                return False, None

        return True, parsed

    def _validate_memory_agent_extracted(self, parsed: Dict, agent_type: str) -> Tuple[bool, Optional[Dict]]:
        """Validate episodic/semantic/procedural agent output in extracted mode.

        Expected operation formats:
          {"action": "ADD", "item_id": 1}
          {"action": "SKIP", "item_id": 2, "reason": "..."}
          {"action": "UPDATE", "old_memory": "...", "item_id": 3}
          {"action": "MERGE", "old_memories": [...], "item_ids": [4, 5]}
        """
        operations = parsed.get('operations', [])

        if not isinstance(operations, list):
            return False, None

        valid_actions = self.VALID_ACTIONS_BY_AGENT_EXTRACTED.get(agent_type, [])

        for op in operations:
            action = op.get('action', '').upper()
            if action not in valid_actions:
                return False, None

            if action == 'ADD':
                if 'item_id' not in op:
                    return False, None
                if not isinstance(op['item_id'], int):
                    return False, None
            elif action == 'UPDATE':
                if 'old_memory' not in op:
                    return False, None
                if 'item_id' not in op:
                    return False, None
                if not isinstance(op['item_id'], int):
                    return False, None
            elif action == 'MERGE':
                if 'old_memories' not in op:
                    return False, None
                item_ids = op.get('item_ids', [])
                if not isinstance(item_ids, list) or not item_ids:
                    return False, None
            elif action == 'SKIP':
                if 'item_id' not in op:
                    return False, None

        return True, parsed
