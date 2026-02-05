# corpus.py
# ----------------------------------------------------------------
# validates a corpus by checking external data related to LWs/cognates
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# feb-2026

import pandas as pd
import json
import os

class CorpusValidator:
    """Validates a corpus by checking external data sources for each language."""
    
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        
        self.path_cognet = "data/external/cognet.tsv"
        self.path_unimorph = "data/external/unimorph_eus.tsv"
        self.path_conloan = "data/external/conloan_ell.tsv"
        
        self.sources = {
            'ast_cognates': set(), # lots of shared words with spanish
            'eu_forms': set(),     # to validate synthetic seeds and integration basque is very integrated, but we can check if the word is attested in unimorph (not a perfect proxy, but a strong signal of integration)
            'el_loans': set()      # to validate synthetic seeds and established loans in greek
        }
        
        self.stats = {}

    def process(self):
        """Counts instances of cognates (from established terms: Asturian), valid integration (of synthetic terms: Euskera)
        and whether established terms found are attested as historical loans in ConLoan (of established terms: Greek)."""
        results = []
        
        self._load()
        self._scan_corpus()
        
        print("> Applying validation logic...")
        
        for (lang, term), stats in self.stats.items():
            term_lower = str(term).lower()
            
            is_wiktionary = any('wiktionary' in t.lower() for t in stats['types'])
            category = 'established' if is_wiktionary else 'new/tech'
            
            AST_is_cognate = 0      # AST
            EU_is_integrated = 0   # EU
            EL_is_historical = 0   # EL
            
            if lang == 'ast':
                if term_lower in self.sources['ast_cognates']:
                    AST_is_cognate = 1
            elif lang == 'eu':
                if term_lower in self.sources['eu_forms']:
                    EU_is_integrated = 1   
            elif lang == 'el':
                if term_lower in self.sources['el_loans']:
                    EL_is_historical = 1
                                        
            results.append({
                'lang': lang,
                'term': term,
                'frequency': stats['freq'],
                'category': category,
                'source_types': ", ".join(stats['types']),
                'is_valid_cognate': AST_is_cognate,       
                'is_valid_integrated': EU_is_integrated, 
                'is_valid_historical': EL_is_historical  
            })
            
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        df.to_csv(self.output_file, index=False)
        print(f">>> Full mined-sentence corpus validation saved to: {self.output_file}")
        self._print(df) 
        
        return df

# -----------------------------------------------------------------------------------------

    def _scan_corpus(self):
        if not os.path.exists(self.input_file):
            print(f"(!) > Corpus file not found: {self.input_file}")
            return

        print(f"> Scanning full corpus: {self.input_file} ...")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    row = json.loads(line)
                    lang = row.get('lang')
                    term = row.get('term')
                    t_type = row.get('type', 'unknown')
                    
                    if not lang or not term: continue
                    
                    key = (lang, term)
                    
                    if key not in self.stats:
                        self.stats[key] = {'freq': 0, 'types': set()}
                    
                    self.stats[key]['freq'] += 1
                    self.stats[key]['types'].add(t_type)
                    
                except json.JSONDecodeError:
                    continue
        print(f"> Found {len(self.stats)} unique term entries.")
        

    def _print(self, df):
        print("\n--- Full mined term corpus ---")
        
        # ** ASTURIAN ** differentiating established borrowings from cognates
        ast_est = df[(df['lang'] == 'ast') & (df['category'] == 'established')]
        if not ast_est.empty:
            cog_count = ast_est['is_valid_cognate'].sum()
            print(f"> Asturian: established LWs that are Latin and Spanish cognates: {cog_count}/{len(ast_est)} ({cog_count/len(ast_est):.1%})")

        # ** EUSKERA ** how many synth seeds are actually integrated and attested in data
        eu_tech = df[(df['lang'] == 'eu') & (df['category'] == 'new/tech')]
        if not eu_tech.empty:
            int_count = eu_tech['is_valid_integrated'].sum()
            print(f"> Euskera: established tech terms found in Unimorph: {int_count}/{len(eu_tech)} ({int_count/len(eu_tech):.1%})")
            
        # ** GREEK ** established terms are recorded historical loans
        el_est = df[(df['lang'] == 'el') & (df['category'] == 'established')]
        if not el_est.empty:
            hist_count = el_est['is_valid_historical'].sum()
            print(f"> Greek: established LWs validated by ConLoan: {hist_count}/{len(el_est)} ({hist_count/len(el_est):.1%})")
        # contrast with new loans
        el_tech = df[(df['lang'] == 'el') & (df['category'] == 'new/tech')]
        if not el_tech.empty:
            hist_tech_count = el_tech['is_valid_historical'].sum()
            print(f"> Greek: new tech terms found in ConLoan: {hist_tech_count}/{len(el_tech)} ({hist_tech_count/len(el_tech):.1%})")
            
    def _load(self):
        """Loads external validation datasets into memory."""
        print("> Loading external dictionaries...")

        # ** COGNET ** (Asturian -> Latin/Spanish Cognates)
        try:
            with open(self.path_cognet, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        l1, t1, l2 = parts[1], parts[2], parts[3]
                        # We capture Asturian terms that share a root with Spanish, Latin, or Iberian
                        if l1 == 'ast' and l2 in ['spa', 'lat', 'xib']:
                            self.sources['ast_cognates'].add(t1.lower())
                            
            print(f" > Asturian: loaded {len(self.sources['ast_cognates'])} cognates from CogNet.")
        except FileNotFoundError:
            print(f"(!) > Warning: CogNet file not found at {self.path_cognet}")

        # ** BASQUE ** (morphology validation)
        try:
            with open(self.path_unimorph, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        form = parts[1] # inflection
                        self.sources['eu_forms'].add(form.lower())
            print(f"> Basque: loaded {len(self.sources['eu_forms'])} inflected forms from Unimorph.")
        except FileNotFoundError:
            print(f"(!) > Warning: Unimorph file not found at {self.path_unimorph}")

        # ** GREEK ** established loans
        try:
            with open(self.path_conloan, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        term = parts[0]
                        self.sources['el_loans'].add(term.lower())
            print(f"> Greek: loaded {len(self.sources['el_loans'])} historical loans from ConLoan.")
        except FileNotFoundError:
            print(f"(!) > Warning: ConLoan file not found at {self.path_conloan}")