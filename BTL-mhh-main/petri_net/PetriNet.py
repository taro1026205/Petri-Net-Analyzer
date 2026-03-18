import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Optional

class PetriNet:
    def __init__(
        self,
        place_ids: List[str],
        trans_ids: List[str],
        place_names: List[Optional[str]],
        trans_names: List[Optional[str]],
        I: np.ndarray,   
        O: np.ndarray, 
        M0: np.ndarray
    ):
        self.place_ids = place_ids #Danh sách ID của các place 
        self.trans_ids = trans_ids #Danh sách ID của các transition
        self.place_names = place_names #Tên hiển thị của các place
        self.trans_names = trans_names #Tên hiển thị của các transition
        self.I = I #Ma trận đầu vào
        self.O = O #Ma trận đầu ra
        self.M0 = M0 #Marking ban đầu

    @classmethod
    def from_pnml(cls, filename: str) -> "PetriNet":
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # Xử lý namespace (thường PNML có namespace, cần lấy nó ra để find)
        # Ví dụ: {http://www.pnml.org/version-2009/grammar/pnml}net
        ns = {}
        if '}' in root.tag:
            ns_url = root.tag.split('}')[0].strip('{')
            ns = {'pnml': ns_url}
        
        # Helper function để tìm thẻ có namespace
        def find_all(element, tag):
            if ns:
                return element.findall(f".//pnml:{tag}", ns)
            return element.findall(f".//{tag}")
            
        def find_text(element, tag):
            # Tìm text trong thẻ con (ví dụ name/text)
            if ns:
                node = element.find(f"pnml:{tag}", ns)
            else:
                node = element.find(f"{tag}")
            return node.text if node is not None else None

        # 1. Thu thập Places
        place_ids = []
        place_names = []
        initial_markings = []
        place_id_map = {} # Map ID -> Index
        
        xml_places = find_all(root, 'place')
        for idx, pl in enumerate(xml_places):
            pid = pl.get('id')
            place_ids.append(pid)
            place_id_map[pid] = idx
            
            # Lấy tên (nếu có)
            name_tag = find_all(pl, 'name')
            pname = None
            if name_tag:
                text_tag = find_text(name_tag[0], 'text')
                pname = text_tag
            place_names.append(pname)
            
            # Lấy initialMarking
            init_tag = find_all(pl, 'initialMarking')
            marking = 0
            if init_tag:
                try:
                    text_val = find_text(init_tag[0], 'text')
                    marking = int(text_val) if text_val else 0
                except ValueError:
                    marking = 0
            initial_markings.append(marking)

        # 2. Thu thập Transitions
        trans_ids = []
        trans_names = []
        trans_id_map = {} # Map ID -> Index
        
        xml_trans = find_all(root, 'transition')
        for idx, tr in enumerate(xml_trans):
            tid = tr.get('id')
            trans_ids.append(tid)
            trans_id_map[tid] = idx
            
            # Lấy tên
            name_tag = find_all(tr, 'name')
            tname = None
            if name_tag:
                text_tag = find_text(name_tag[0], 'text')
                tname = text_tag
            trans_names.append(tname)

        # 3. Khởi tạo ma trận và vector
        num_places = len(place_ids)
        num_trans = len(trans_ids)
        
        I = np.zeros((num_trans, num_places), dtype=int)
        O = np.zeros((num_trans, num_places), dtype=int)
        M0 = np.array(initial_markings, dtype=int)

        # 4. Duyệt Arcs để điền ma trận
        xml_arcs = find_all(root, 'arc')
        for arc in xml_arcs:
            source = arc.get('source')
            target = arc.get('target')
            
            # Lấy trọng số (inscription), mặc định là 1
            weight = 1
            insc_tag = find_all(arc, 'inscription')
            if insc_tag:
                try:
                    text_val = find_text(insc_tag[0], 'text')
                    weight = int(text_val) if text_val else 1
                except ValueError:
                    weight = 1
            
            # Kiểm tra loại cung: Place -> Transition (Input Matrix)
            if source in place_id_map and target in trans_id_map:
                p_idx = place_id_map[source]
                t_idx = trans_id_map[target]
                I[t_idx, p_idx] = weight
                
            # Kiểm tra loại cung: Transition -> Place (Output Matrix)
            elif source in trans_id_map and target in place_id_map:
                t_idx = trans_id_map[source]
                p_idx = place_id_map[target]
                O[t_idx, p_idx] = weight

        return cls(place_ids, trans_ids, place_names, trans_names, I, O, M0)

    def __str__(self) -> str:
        s = []
        s.append("Places: " + str(self.place_ids))
        s.append("Place names: " + str(self.place_names))
        s.append("\nTransitions: " + str(self.trans_ids))
        s.append("Transition names: " + str(self.trans_names))
        s.append("\nI (input) matrix:")
        s.append(str(self.I))
        s.append("\nO (output) matrix:")
        s.append(str(self.O))
        s.append("\nInitial marking M0:")
        s.append(str(self.M0))
        return "\n".join(s)


