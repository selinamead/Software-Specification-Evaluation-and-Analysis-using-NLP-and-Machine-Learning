from __future__ import print_function

import re
import os
import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import nltk
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from plotly.offline import  plot
import plotly.graph_objs as go
from textblob_de import TextBlobDE
from stop_words import get_stop_words
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


# ===========================================================================================================================

class NLP_Features():

     def __init__(self):

          self.df=pd.DataFrame()
          self.weakwords = [ '!', '######', '?', '__', 'ab', 'aber', 'abgestimmt', 'abgestimmte', 'abgestimmtem', 'abgestimmten', 'abgestimmter', 'abgestimmtes',
                         'absolut', 'absprache', 'absprachen', 'abstimmen', 'abstimmung', 'abstimmungspflichtig', 'abteilung', 'abzustimmen', 'achten', 'ähnlich',
                         'aktuell', 'akustikverhalten', 'alle', 'allerdings', 'allgemein', 'allgemein gültig', 'allzu', 'als ob', 'alternativ', 'alternative',
                         'alternativem', 'alternativen', 'alternativer', 'alternatives', 'andere', 'andererseits', 'andernfalls', 'anders', 'änderungsdokumentation',
                         'anfangs', 'angebot', 'angeboten', 'angemessen', 'anhaltend', 'anlehnung', 'annähernd', 'annahme', 'annahmen', 'anscheinend', 'anzubieten',
                         'anzugeben', 'anzuwenden', 'anzuwendende', 'anzuwendenden', 'arbeitsweise', 'auf keinen Fall', 'aufgezeigt', 'auftragnehmer', 'aufzuzeigen',
                         'augenblicklich', 'augenscheinlich', 'ausarbeiten', 'ausführlich', 'ausgearbeitet', 'ausgegangen', 'ausgehen', 'ausgewiesen', 'ausnahmslos',
                         'ausnahmsweise', 'ausreichen', 'ausreichend', 'ausreichende', 'ausreichendem', 'ausreichenden', 'ausreichender', 'ausreichendes',
                         'aussagekräftig', 'außen', 'außerdem', 'außerhalb', 'außerordentlich', 'ausweisen', 'auszuarbeiten', 'auszuführen', 'auszugehen',
                         'auszuweisen', 'bald', 'beachten', 'beachtet', 'bedarf', 'bedarfsabhängig', 'bedarfsabhängige', 'bedarfsabhängigem', 'bedarfsabhängigen',
                         'bedarfsabhängiger', 'bedarfsweise', 'bedarfsweisem', 'bedarfsweisen', 'bedarfsweiser', 'bedarfsweises', 'bedienbar', 'bedingt', 'bei',
                         'beinahe', 'beispiel', 'beispiele', 'beispielhaft', 'beispielhafte', 'beispielhaftem', 'beispielhaften', 'beispielhafter', 'beispielhaftes',
                         'beispieltext', 'belieferung', 'berechnen', 'berechnet', 'bereitgestellt', 'berücksichtigen', 'berücksichtigt', 'besondere', 'besonderen',
                         'besonderes', 'besonders', 'besser', 'bestätigen', 'bestätigt', 'beste', 'bestimmt', 'bestimmungsgemäß', 'bestmöglich', 'bestmögliche',
                         'bestmöglichem', 'bestmöglichen', 'bestmöglicher', 'bestmögliches', 'bevorzugt', 'bewertungsindex', 'bi', 'bi8', 'bieten', 'bieter', 'bisher',
                         'bosch', 'bringen', 'ca.', 'circa', 'cirka', 'coc', 'continental', 'damals', 'danach', 'daneben', 'dann', 'darauf', 'daraufhin', 'darf',
                         'darstellen', 'darstellung', 'darüber hinaus', 'darzustellen', 'das selbe', 'dasselbe', 'demnächst', 'denkbar', 'denn', 'der selbe',
                         'der selben', 'dereinst', 'derselbe', 'deutlich', 'dicht', 'die selben', 'dieselbe', 'dieselben', 'diesseits', 'doch', 'dort', 'dorther', 
                         'dorthin', 'dp44', 'draußen', 'dringend', 'dringende', 'dringendem', 'dringenden', 'dringender', 'dringendes', 'drinnen', 'durchaus',
                         'durchführen', 'durchgängig', 'durchgeführt', 'durchgehend', 'durchweg', 'durchzuführen', 'dürfen', 'ea-', 'ebenfalls', 'eg-', 'ehedem',
                         'ehemalige', 'ehemals', 'eher', 'eigentlich', 'eilends', 'ein bisschen', 'ein paar', 'ein wenig', 'eindeutig', 'eine Weile', 'einerseits',
                         'einfach', 'einfache', 'einfachem', 'einfachen', 'einfacher', 'einfaches', 'einigermaßen', 'einmal', 'einmalig', 'einst', 'einstmals',
                         'einzeln', 'einzuschätzen', 'elementar', 'empfehlen', 'empfehlung', 'empfohlen', 'endgültig', 'endgültige', 'endgültigem', 'endgültigen',
                         'endgültiges', 'endlich', 'eng', 'enorm', 'entsprechend', 'entsprechende', 'entsprechendem', 'entsprechenden', 'entsprechender',
                         'entsprechendes', 'entwicklungshandbuch', 'entwicklungsumfang', 'erbringen', 'erfassen', 'erfassung', 'ergänzt', 'ermitteln', 'ermittelt',
                         'erprobungshandbuch', 'erprobungshandbücher', 'erscheint', 'erstaunlich', 'erstellen', 'erstellung', 'essentiell', 'etliche', 'etwa',
                         'etwa wie', 'etwas', 'etwelche', 'eventuell', 'evtl.', 'extra', 'ezk', 'fa.', 'fabelhaft', 'fachabteilung', 'fachbereich', 'fachstelle',
                         'fahrzeugseitig', 'fahrzeugseitige', 'fahrzeugseitigem', 'fahrzeugseitiger', 'fahrzeugseitiges', 'falsch', 'falsche', 'falschem', 'falschen',
                         'falscher', 'falsches', 'falschverbau', 'false', 'fast', 'ferner', 'festzulegen', 'firma', 'fortan', 'fortlaufend', 'fortschrittlich', 'früh',
                         'führen', 'funktionalität', 'für den Fall', 'furchtbar', 'gängig', 'ganz', 'gar', 'geachtet', 'gebräuchlich', 'gebräuchliche', 'gebräuchlichem',
                         'gebräuchlichen', 'gebräuchlicher', 'gebräuchliches', 'geeignet', 'geeignete', 'geeignetem', 'geeigneten', 'geeigneter', 'geeignetes',
                         'gegebenenfalls', 'gegen', 'gegenwärtig', 'gelegentlich', 'gemeinsam', 'gemeinsame', 'gemeinsamen', 'gemeinsamer', 'gemeinsames', 'genau',
                         'generell', 'genug', 'geomet', 'gerade so', 'geräuschoptimiert', 'geräuschoptimierte', 'geräuschoptimiertem', 'geräuschoptimierten',
                         'geräuschoptimierter', 'geräuschoptimiertes', 'geräuschoptimierung', 'gering', 'geringe', 'geringem', 'geringen', 'geringer', 'geringere',
                         'geringes', 'geringst', 'geringste', 'geringstem', 'geringsten', 'geringster', 'geringstes', 'gesamt', 'gesamtverantwortlich',
                         'gesamtverantwortung', 'gesendet', 'gesondert', 'gestern', 'gestrige', 'gewährleisten', 'gewährleistet', 'gewaltig', 'gewicht',
                         'gewichtoptimiert', 'gewichtsoptimiert', 'gewichtsoptimierte', 'gewichtsoptimiertem', 'gewichtsoptimierten', 'gewichtsoptimierter',
                         'gewichtsoptimiertes', 'gewichtsoptimierung', 'gewichtsoptimierungen', 'gewichtsoptimum', 'gewisse', 'gewohnt', 'ggf', 'gleich',
                         'gleichzeitig', 'gleichzeitige', 'gleichzeitigem', 'gleichzeitigen', 'gleichzeitiger', 'gleichzeitiges', 'groß', 'große', 'großem',
                         'großen', 'großer', 'größer', 'großes', 'größtenteils', 'größtmöglich', 'größtmögliche', 'größtmöglichem', 'größtmöglichen',
                         'größtmöglicher', 'größtmögliches', 'größtmöglichst', 'größtmöglichste', 'größtmöglichstem', 'größtmöglichsten', 'größtmöglichster',
                         'größtmöglichstes', 'grundsätzlich', 'günstig', 'günstigem', 'günstiger', 'günstiges', 'gut', 'halbwegs', 'halt', 'häufig', 'hauptsächlich',
                         'heranzuziehen', 'hernach', 'hervorzuheben', 'heutigentags', 'heutzutage', 'hier', 'hierauf', 'hierhin', 'hin und wieder', 'hinten',
                         'hinterher', 'hk', 'höchst', 'höchstens', 'höchstwahrscheinlich', 'hoffentlich', 'höheren wert', 'höherer wert', 'im Allgemeinen',
                         'im Augenblick', 'im Moment', 'indes', 'indessen', 'informieren', 'informiert', 'innerhalb', 'insbesondere', 'intuitiv', 'inzwischen',
                         'irgend', 'irgendetwas', 'irgendwelche', 'irgendwer', 'irgendwo', 'irgendwoher', 'irgendwohin', 'ja', 'jahr', 'jahre', 'jahren',
                         'jährlich', 'je nachdem', 'jedenfalls', 'jeder', 'jedermann', 'jederzeit', 'jedoch', 'jemals', 'jenseits', 'jetzt', 'kann', 'kaum',
                         'kein', 'keinesfalls', 'klassisch', 'klein', 'knapp', 'kolossal', 'können', 'könnte', 'könnten', 'konzeptwettbewerb', 'konzeptwettbewerber',
                         'konzeptwettbewerbes', 'korrekt', 'korrekte', 'korrektem', 'korrekten', 'korrekter', 'korrektes', 'kosten', 'kosteneinfluss', 'kostengünstig',
                         'kostenpotential', 'kostenpotenzial', 'kostenpotenziale', 'kostenreduktion', 'kurz', 'kürzlich', 'kurzzeitig', 'kurzzeitige', 'kurzzeitigem',
                         'kurzzeitigen', 'kurzzeitiger', 'kurzzeitiges', 'lagern', 'landläufig', 'lang', 'länger', 'längere', 'längeren', 'längerer', 'längeres', 'langsam',
                         'längst', 'längstens', 'laufe', 'laufend', 'laufruhe', 'laut', 'leicht', 'leichte', 'leichtem', 'leichten', 'leichtere', 'leichteren', 'leise',
                         'letztens', 'lieferant', 'lieferanten', 'lieferantenauswahl', 'lieferantenfestlegung', 'lieferumfang', 'lieferumfänge', 'lieferung', 'links',
                         'ma-', 'mahle', 'man', 'manche', 'manchmal', 'mangelhaft', 'masse', 'mäßig', 'maximieren', 'maximiert', 'maximierte', 'maximiertem', 'maximierten',
                         'maximierter', 'mehr', 'mehr oder minder', 'mehrere', 'mehrfach', 'mehrmals', 'meist', 'meistens', 'melden', 'messen', 'minder', 'minimal',
                         'minimale', 'minimalen', 'minimaler', 'minimieren', 'minimiert', 'minimierte', 'minimiertem', 'minimierten', 'minimierter', 'minimiertes',
                         'mitgeteilt', 'mitteilen', 'mitunter', 'mitzuliefern', 'mk', 'möchte', 'mögen', 'möglich', 'mögliche', 'möglichem', 'möglichen', 'möglicher',
                         'möglicherweise', 'mögliches', 'möglichkeit', 'möglichst', 'momentan', 'monat', 'monate', 'monatlich', 'morgen', 'müsste', 'müssten', 'mustertext',
                         'nachdem', 'nachher', 'nächstens', 'nachts', 'nachweisen', 'nachzuweisen', 'nahezu', 'nebenbei', 'neu', 'neuartig', 'neulich', 'nicht gefordert',
                         'nichts', 'nie', 'niemals', 'niemand', 'nimmermehr', 'nirgends', 'nirgendwo', 'nirgendwohin', 'nominierung', 'notbetrieb', 'nötig', 'nötigenfalls',
                         'notlauf', 'nur', 'oben', 'offensichtlich', 'oft', 'öfter', 'öftere', 'öfterem', 'öfteren', 'öfterer', 'öfteres', 'öfters', 'optimal', 'optimale',
                         'optimalem', 'optimalen', 'optimaler', 'optimales', 'optimieren', 'optimiert', 'optimierte', 'optimiertem', 'optimierten', 'optimierter',
                         'optimiertes', 'optimierung', 'optimierungspotential', 'optimierungspotenzial', 'optimum', 'option', 'optional', 'optionale', 'optionalem',
                         'optionalen', 'optionaler', 'optionales', 'optionen', 'optisch', 'orientierung', 'orientierungswert', 'paket', 'parallel', 'pauschal', 'pflichttext',
                         'phantastisch', 'pkgo', 'plausibel', 'präsentation', 'präsentieren', 'preis', 'preislich', 'primär', 'primäre', 'primärem', 'primären', 'primärer',
                         'primäres', 'prinzipiell', 'priorität', 'problem', 'prüfen', 'puffer', 'rechts', 'redundanz', 'referenzenchecker', 'regelmäßig', 'regelmäßige',
                         'regelmäßigem', 'regelmäßigen', 'regelmäßiger', 'regeln der technik', 'reichlich', 'relativ', 'relevant', 'relevante', 'relevantem', 'relevanter',
                         'relevantes', 'reserve', 'rhythmus', 'riesig', 'rund', 'sämtliche', 'schätzungsweise', 'scheinbar', 'schicht', 'schichtweise', 'schicken',
                         'schlecht', 'schließlich', 'schnell', 'schnelle', 'schnellem', 'schnellen', 'schneller', 'schnellstens', 'schön', 'schrecklich', 'schwer',
                         'schwerlich', 'sehr', 'seinerzeit', 'seite', 'seiten', 'seitens', 'selbsterklärend', 'selten', 'senden', 'separat', 'serie', 'sicher', 'sicherlich',
                         'sichtprüfung', 'so', 'sofort', 'sogar', 'sogleich', 'solche', 'soll', 'sollen', 'sollte', 'sollten', 'sollwert', 'sollwerte', 'somit', 'sonstige',
                         'sorgfältig', 'sorgfaltspflicht', 'sozusagen', 'spät', 'später', 'spätere', 'späteren', 'standardisiert', 'stark', 'stets', 'störend', 'störende',
                         'störendem', 'störenden', 'störender', 'störgeräusch', 'störgeräuschfrei', 'störgeräuschfreiheit', 'stufe', 'stufung', 'stündlich', 'systemlieferant',
                         'systemlieferanten', 't.b.c.', 't.b.d.', 'ta-', 'tag', 'tagesweise', 'täglich', 'tägliche', 'täglichem', 'täglichen', 'täglicher', 'tägliches',
                         'tagsüber', 'tbc', 'tbd', 'team', 'teiligkeit', 'teils', 'teilweise', 'text', 'total', 'typisch', 'typische', 'typischen', 'typischer',
                         'typischerweise', 'typisches', 'u. a.', 'u.a.', 'u.s.w.', 'u.U.', 'überall', 'überallher', 'überaus', 'überhaupt', 'übermorgen', 'üblich',
                         'übrige', 'umfang', 'umgehend', 'unabdingbar', 'unbedingt', 'unbeträchtlich', 'und', 'und wann', 'ungeeignet', 'ungeeignete', 'ungeeignetem',
                         'ungeeigneten', 'ungeeigneter', 'ungeeignetes', 'ungefähr', 'ungemein', 'ungezählt', 'universal', 'universell', 'unlängst', 'unmerklich',
                         'unter Umständen', 'unterdessen', 'unterstützung', 'usw', 'usw.', 'va', 'varianten', 'verantworten', 'verantwortet', 'verantwortlich',
                         'verantwortung', 'verbessern', 'verbessert', 'verbesserte', 'verbessertem', 'verbessertes', 'verblüffend', 'vereinbaren', 'vereinbarung',
                         'vereinfachen', 'vereinzelt', 'verfahrensanweisung', 'verfügung', 'vergabe', 'vergleichbar', 'vergleichbare', 'vergleichbarem', 'vergleichbaren',
                         'vergleichbarer', 'vergleichbares', 'vermeidbar', 'vermeidbare', 'vermeidbarem', 'vermeidbaren', 'vermeiden', 'vermutlich', 'versenden',
                         'versprechen', 'verständlich', 'versuchsrichtlinie', 'viel', 'viele', 'vielfach', 'vielleicht', 'vielmal', 'vierteljährlich', 'vollendet',
                         'völlig', 'vollkommen', 'vollständig', 'vorerst', 'vorgeschlagen', 'vorgestellt', 'vorhalt', 'vorhin', 'vorläufig', 'vorläufige', 'vorläufigem',
                         'vorläufigen', 'vorläufiger', 'vorläufiges', 'vorn', 'vorne', 'vorschlag', 'vorschlagen', 'vorsgechlagen', 'vorzugsweise', 'vorzuschlagen',
                         'vorzustellen', 'vr', 'wahnsinnig', 'während', 'währenddessen', 'wahrscheinlich', 'weise', 'weit', 'weitaus', 'weitem', 'weitere', 'weiteren',
                         'weiterhin', 'weiters', 'werkstattbetrieb', 'werkstattüblich', 'wesentlich', 'wettbewerb', 'wettbewerber', 'wichtig', 'wichtige', 'wichtigem',
                         'wichtigen', 'wichtiger', 'wichtiges', 'wie wenn', 'winzig', 'wird', 'wirklich', 'woche', 'wochen', 'wöchentlich', 'wöchentliche', 'wöchentlichem',
                         'wöchentlichen', 'wöchentlicher', 'wöchentliches', 'wochenweise', 'wohl', 'wollen', 'womöglich', 'xx', 'xxx', 'xy', 'xyz', 'z.T.', 'zahllos',
                         'zahlreich', 'zak', 'zeitiges', 'zeitlebens', 'zeitweise', 'ziemlich', 'zirka', 'zu', 'zu meist', 'zudem', 'zuerst', 'zufriedenstellend', 'zugestimmt',
                         'zugleich', 'zuletzt', 'zulieferer', 'zunächst', 'zusatz', 'zusätzlich', 'zuständig', 'zustimmen', 'zuvor', 'zuweilen', 'zuweisen', 'zuzustimmen',
                         'zwingend', 'zwischendurch', 'zyklisch', 'zyklische', 'zyklischem', 'zyklischen', 'zyklischer']
          
          self.stemmer = SnowballStemmer("german")

     
     #============================================================================================================#
     ''' 
     Functions to extract nlp features from the requirements 

     '''
     
     """ selects the number of sentences in one requirement and takes the greater number from nltk or spacy """
     def select_sentences(self, row, nb="n"):
          # check if nltk detects less sentences than spacy
          if (row['sentence_nb_by_nltk']<=row['sentence_nb_by_nlp']):
               if nb == "y":
                    return row['sentence_nb_by_nltk']
               else:
                    return row['sentences_by_nltk']
          # if the module nltk detects more sentences than spacy module
          else:
               if nb == "y":
                    return row['sentence_nb_by_nlp']
               else:
                    return row['sentences_by_nlp']

     """ POS Tagging of requirements """
     def tag_sentence(self,nlp,sentence):
          return [(w.text, w.tag_, w.pos_) for w in nlp(sentence)]

     """ function that returns the number of syllables per words. """
     # def _syllables(word):
     #      syllable_count = 0
     #      vowels = 'aeiouy'
     #      if word[0] in vowels:
     #           syllable_count += 1
     #      for index in range(1, len(word)):
     #           if word[index] in vowels and word[index - 1] not in vowels:
     #                syllable_count += 1
     #      if word.endswith('e'):
     #           syllable_count -= 1
     #      if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
     #           syllable_count += 1
     #      if syllable_count == 0:
     #           syllable_count += 1

     #      return syllable_count

     def compute_SPW(self, text):
          syllable = 0
          vowels = ['a','e','i','o','u','y','ä','ü','ö','-']
          diphtong = ['eu','au','ei','äu','io','ai','oi','ui']
          for word in text.split():
               count=0
          for vowels in ['a','e','i','o','y','ä','ü','ö','-']:
               count+=word.count(vowels)
          for diphtong in ['eu','au','ei','äu']:
               count-=word.count(diphtong)
          if count == 0:
               count += 1
               syllable+=count
          
          # calculate the syllable mean (total syllable divided by number of words in the sentence)
          return syllable/(len(text.split())*1.0)

     def count_punctuation(self,attribute):
          i = 0
          for w in range(len(attribute)):
          # if it is tagged with "$(" (<=> punctuation tag)
               if attribute[w][1]=="$(":
                    i+=1
          return i

     def count_comma(self,attribute):
          i = 0
          for w in range(len(attribute)):
               if attribute[w][1]=="$,":
                    i+=1
          return i

     def count_weird_words(self, attribute):
          i = 0
          for w in range(len(attribute)):
               if attribute[w][2]=="X":
                    i+=1
          return i

     """ Search for specific words in the requirements. When word is beispiel or circa, take abbreviation into account """
     def search_specific_words(self, attribute, search_word):
          if search_word=="beispiel":
               word = ["z.b.","zb","zum beispiel","zb."]
          elif search_word=="circa":
               word = ["circa","ca","ca."]
          else: word = [search_word.lower()]
          i = 0
          for w in range(len(attribute)):
               if attribute[w][0].lower() in word:
                    i+=1
          return i

     """ function that checks the presence of "und" or "oder" in the requirement (double counting with CONJ?)"""
     def count_Copulative_Disjunctive_terms(self, requirement):
          disj_list = ["und","oder"]
          i = 0
          for w in requirement.split():
               if w.lower() in disj_list:
                    i+=1
          return i

     """ Checks for the presence of "minimum" or "maximum" and their derivated forms"""
     def check_max_min_presence(self, requirement):
          # list of derivated forms of minimum and maximum
          maxmin_list = ["max","maximum","min","minimum","max.","min.","min-/max","maximal","maximale","maximalen","maximaler","minimal","minimale","minimalen","minimaler"]
          presence = "no"
          for w in requirement.split():
               if w.lower() in maxmin_list:
                    presence = "yes"

          return presence

     """ function that check if some logical conjunction are present in the requirement """
     def time_logical_conj(self,attribute):
          # list of logical conjunction taken into account
          conj_list = ["während","sobald","bis","innerhalb","bei","wenn","gemäß","falls","bzw."]
          i = 0
          for w in range(len(attribute)):
               if attribute[w][0].lower() in conj_list:
                    i+=1
          return i


     def search_measurements_indicators(self, attribute):

          scale_1 = ["sec","sekund","stunde","h","minut","Grad","%","n","km","km/h","pa","rad/sec","/s","°/s","cm","m/s","m/s^2"]
          #scale_2 = ["sec","sekund","stunde","h","minut","min","Grad","%","n","km","km/h","pa","rad/sec","/s","°/s","cm","m/s","m/s^2"]
          scale_2 = ["%","rad/sec","/s","°/s","cm","m/s","m/s^2"]
          presence = False
          for w in range(len(attribute)):
               if attribute[w][0].lower() in scale_1:
                    presence = True
          for s in scale_2:
               if s in attribute[w][0].lower():
                    presence = True

          if presence:
               return "yes"
          else:
               return "no"

     def search_numerical_value(self, attribute):
          i = 0
          for w in range(len(attribute)):
               if attribute[w][2]=="NUM":
                    i+=1
          return i

     def get_weakwords(self):

          weakword_list_1 = "data/Weakwords/WWF_Word-Liste.xlsx"
          weakword_list_2 = "data/Weakwords/Dict_WW_Studio_input_MASTER_New.xlsx"
          df1 = pd.read_excel(weakword_list_1)
          df2 = pd.read_excel(weakword_list_2)
          weakwords_list = list(df1["Deutsch"]) + list(df2["Lemma"])
          weakwords_list_str = [str(w) for w in weakwords_list]
          weakwords_list_lower = [w.lower() for w in weakwords_list_str]

          self.weakwords = weakwords_list_lower

     def count_weakwords_from_tags(self, attribute):
          i = 0
          for w in range(len(attribute)):
               if attribute[w][0].lower() in self.weakwords:
                    i+=1
               else:
                    w_stem = self.stemmer.stem(attribute[w][0].lower())
                    if w_stem in self.weakwords:
                         i+=1
          return i

     def count_weakwords_from_sentence(self, sentence):
          i = 0
          weak = []
          words = sentence.split()
          for w in words:
               if w.lower() in self.weakwords:
                    i+=1
                    weak.append(w)
          return i, weak

     def count_weakwords_from_lemma(self, attribute, stemmer):
          i = 0
          weak = []
          for w in range(len(attribute)):
               if attribute[w][0].lower() in self.weakwords:
                    i+=1
                    weak.append(attribute[w][0].lower())
               else:
                    w_stem = stemmer.stem(attribute[w][0].lower())
                    if w_stem in self.weakwords:
                         i+=1
                         weak.append(w_stem)
          return i, weak

     """ Function to determine if a sentence is written in passive form
          # if at least one past participe and one auxiliary ("werden", "wird", "worden" "wurde") = passive sentence
          attribute corresponds to the requirements processed by nlp = (word_, tag_, pos_)
     """
     def passive_detection(self, attribute):
          answer = "no"
          # suppose first there is no past participe
          PP = False
          werden_forms = ["werden","wird","worden", "wurde"]
          for w in range(len(attribute)):
               if attribute[w][1]=="VVPP":
                    PP = True
          for w in range(len(attribute)):
          # for each word, look at the ones that belongs to the werden_forms list and that have a "AUX" position
               if (attribute[w][2]=="AUX" and (attribute[w][0] in werden_forms) and PP):
                    # if an AUX werden and a past participe...then passive form
                    answer = "yes"

          return answer

     """ checks if the first word of the requirement is an auxiliary """
     def aux_1st(self,attribute):
          if attribute[0][2]=="AUX":
               return "yes"
          else:
               return "no"


     """ function that counts how many subordinate conjunctions: 
         (e.g. after, although, as, because, before, even if, even though, if, in order that, once, provided that, 
         rather than, since, so that, than, that, though, unless, until, when, whenever, where, whereas, wherever, 
         whether, while, why) are present in the requirement 
     """
     def count_subordinate_conjunction(self, attribute):
          nb_sc=sum(attribute[x][2]=="SCONJ" for x in range(len(attribute)))
          return nb_sc


     """ function that counts how many coordination conjunctions or comparison conjunctions 
         (und, oder, als, bzw., bis, oder) are present in the requirement
     """
     def count_comp_coor_conjunction(self, attribute):
          nb_cc=sum(attribute[x][2]=="CONJ" for x in range(len(attribute)))
          return nb_cc

     """ counts how many verbs are present in requirement """
     def count_verb(self, attribute):
          nb_vb=sum(attribute[x][2]=="VERB" for x in range(len(attribute)))
          return nb_vb

     """ function that counts how many auxiliaries are present in the requirement """
     def count_aux(self, attribute):
          nb_aux=sum(attribute[x][2]=="AUX" for x in range(len(attribute)))
          return nb_aux

     """ function that counts how many times the word "werden" or its conjugate forms appear in a requirement """
     def count_werden(self, text):
          count=0
          if re.search("wird", text,re.I):
               count+=len(re.findall("wird", text,re.I))
          if re.search("werden", text,re.I):
               count+=len(re.findall("werden", text,re.I))
          return count

     """ Finds requirements containing mussen, darfen """
     def contain_Muss_Darf_nicht(self, ps, attribute):
        # possible forms for müssen
        muessen=['muss','musst']
        # possible forms for dürfen
        duerfen=['darf','durf','durft']                                     
        # stem each word in requirement
        tokens=[ps.stem(w) for w in attribute.split()]
        presence = "no"
        for i in range(len(tokens)-1):
          if ((tokens[i] in muessen) or (tokens[i] in duerfen) or (tokens[i] in duerfen and (tokens[i+1]=="nicht" or tokens[i+1]=="maximal" or tokens[i+1]=="höchstens" ))):
                presence="yes"
                return presence

     def entities_label(self, text):
          # Find named entities, phrases and concepts
          if len(text.ents)!= 0:
               return [(x.text,x.label_) for x in text.ents]
          else:
               return text.ents

     # ============================================================================================================== #

     '''
     A function to extract features from requirements and create a dataframe
     '''
     def extract_features(self, Req_list, score_target, export = True, corpal = True):

          nlp = spacy.load('de_core_news_sm')
          stemmer = SnowballStemmer("german")
          stop = stopwords.words('german')
          features = pd.DataFrame()
          # create first column of dataframe by allocating requirement list to it; one requirement per line
          features['req']=Req_list
          # get text, tag_ and pos_ attributes for each word
          features['req_nlp']=features['req'].apply(lambda x:nlp(x))
          features['tags']= features['req_nlp'].apply(lambda x: [(w.text, w.tag_, w.pos_) for w in x])

          # Analysis using NLTK
          # Split sentences then count number in each requirement
          features['sentences_by_nltk']=features['req'].apply(lambda x:nltk.sent_tokenize(x,'german'))
          features['sentence_nb_by_nltk']=features['req'].apply(lambda x:len(nltk.sent_tokenize(x,'german')))
          # analysis with spacy
          features['sentences_by_nlp']=features['req_nlp'].apply(lambda x:[sent.string.strip() for sent in x.sents])
          features['sentence_nb_by_nlp']=features['req_nlp'].apply(lambda x:len([sent.string.strip() for sent in x.sents]))

          # number of sentences per requirement
          features['sentences']=features.apply(lambda x: self.select_sentences(x),axis=1)
          features['sentences_nb']=features.apply(lambda x: self.select_sentences(x,"y"),axis=1)
          features['sentences_tagged']=features['sentences'].apply(lambda x: [self.tag_sentence(nlp,w) for w in x])

          # Calculating Readability-Index
          # words in requirement
          features['words_nb']=features['req'].apply(lambda x:len(x.split()))
          # words per sentence
          features['WPS']=features['words_nb']/features['sentences_nb']
          # syllables per word
          features['SPW']=features['req'].apply(lambda x: self.compute_SPW(x))
          # flesch index
          features['Flesch_Index']=features.apply(lambda x:round((180-x['WPS']-(58.5*x['SPW']))),axis=1)
          # Analyzing punctuation
          features['internal_punctuation'] = features['tags'].apply(lambda x: self.count_punctuation(x))
          features['comma'] = features['tags'].apply(lambda x: self.count_comma(x))
          features['weird_words'] = features['tags'].apply(lambda x: self.count_weird_words(x))

          # Analyzing and counting specific words and list containing words
          features['beispiel'] = features['tags'].apply(lambda x: self.search_specific_words(x,'beispiel'))
          features['circa'] = features['tags'].apply(lambda x: self.search_specific_words(x,'circa'))
          features['wenn'] = features['tags'].apply(lambda x: self.search_specific_words(x,'wenn'))
          features['aber'] = features['tags'].apply(lambda x: self.search_specific_words(x,'aber'))
          features['max_min_presence'] = features['req'].apply(lambda x: self.check_max_min_presence(x))
          features['Nb_of_Umsetzbarkeit_conj'] = features['tags'].apply(lambda x: self.time_logical_conj(x))
          features['measurement_values'] = features['tags'].apply(lambda x: self.search_measurements_indicators(x))
          features['numerical_values'] = features['tags'].apply(lambda x: self.search_numerical_value(x))
          features['polarity'] = features['req'].map(lambda text: TextBlobDE(text).sentiment.polarity)

          # Analyze and create weakwords
          features['weakwords_nb'] = features['tags'].apply(lambda x:self.count_weakwords_from_tags(x))
          features['weakwords_nb2'] = features['req'].apply(lambda x:self.count_weakwords_from_sentence(x))
          features['weakwords_nb2_lemma'] = features['tags'].apply(lambda x: self.count_weakwords_from_lemma(x, stemmer))
          features['difference'] = features['weakwords_nb2_lemma'].apply(lambda x:x[0]) - features['weakwords_nb2'].apply(lambda x:x[0])

          # Analyzing passive and active and auxiliary attributes at the beginning of a requirement
          features['passive_global'] = features['tags'].apply(lambda x: self.passive_detection(x))
          features['passive_per_sentence'] = features['sentences_tagged'].apply(lambda x: [self.passive_detection(s) for s in x])
          features['passive_percent'] = features['passive_per_sentence'].apply(lambda x: (sum([y=="yes" for y in x])/len(x)))
          features['Aux_Start'] = features['tags'].apply(lambda x:self.aux_1st(x))
          features['Aux_Start_per_sentence'] = features['sentences_tagged'].apply(lambda x:[self.aux_1st(s) for s in x])

          # Analyzing conjunctions, verbs and auxiliaries
          features['Sub_Conj']=features['tags'].apply(lambda x:self.count_subordinate_conjunction(x))
          features['Comp_conj']=features['tags'].apply(lambda x:self.count_comp_coor_conjunction(x))
          features['Nb_of_verbs']=features['tags'].apply(lambda x:self.count_verb(x))
          features['Nb_of_auxiliary']=features['tags'].apply(lambda x:self.count_aux(x))
          features['werden']=features['req'].apply(lambda x:self.count_werden(x))

          # same functions as previous block but analysis made for each sentence on one requirement
          features['Sub_Conj_pro_sentece']=features['sentences_tagged'].apply(lambda x:[self.count_subordinate_conjunction(s) for s in x])
          features['Comp_conj_pro_sentence']=features['sentences_tagged'].apply(lambda x:[self.count_comp_coor_conjunction(s) for s in x])
          features['Nb_of_verbs_pro_sentence']=features['sentences_tagged'].apply(lambda x:[self.count_verb(s) for s in x])
          features['Nb_of_auxiliary_pro_sentence']=features['sentences_tagged'].apply(lambda x:[self.count_aux(s) for s in x])
          features['werden_pro_sentence']=features['sentences'].apply(lambda x:[self.count_werden(s) for s in x])

          features['formal_global'] = features['req'].apply(lambda x:self.contain_Muss_Darf_nicht(stemmer,x))
          features['formal_per_sentence'] = features['sentences'].apply(lambda x:[self.contain_Muss_Darf_nicht(stemmer,s) for s in x])
          features['formal_percent'] = features['formal_per_sentence'].apply(lambda x: (sum([y=="yes" for y in x])/len(x)))
          features['entities'] = features['req_nlp'].apply(lambda x:self.entities_label(x))

          # Graphical representation of the vocabulary of requirements corpus
          if corpal:
               self.Corpus_Analysis(Req_list,stop)

          if export:
               my_path = Path(u"/Users/selina/Code/Python/Thesis_Code/Features/" + 'export_features')
               g_Dirpath= os.path.abspath(my_path)
               dataFile = g_Dirpath + '\\' + 'Features_Export.xlsx'
               print ("Create Excel export file: %s"%(dataFile))
               features[0:5000].to_excel(dataFile, index=False)
               print ("\nFeatures_Export XLS-file created and data copied.")

          weakword_al = True
          if weakword_al:
               weakword_analysis = features[['weakwords_nb2','weakwords_nb2_lemma','difference']]
               datafile = "./Generated_Files/" + str(score_target) + "/Weakword_Analysis/" + str(score_target) + ".xlsx"
               weakword_analysis.to_excel(datafile, index=False)
               weakword_analysis2 = features[features['difference']!= 0][["req","tags",'weakwords_nb2','weakwords_nb2_lemma','difference']]
               datafile = "./Generated_Files/" + str(score_target) + "/Weakword_Analysis/" + str(score_target) + ".xlsx"
               weakword_analysis2.to_excel(datafile, index=False)
               print("\n \n")
               # print(features[features['difference']!= 0][["req","tags",'weakwords_nb2','weakwords_nb2_lemma','difference']])
               print("\nFeatures are retrieved and delivered to Data_Preprocessing\n\n")
               # print(features.sentences_tagged)


          return features, features.sentences_tagged
     
     # ======================================================================================================= #
     
     # NOT WORKING
     # """ Reads the "requirement" column of an excel file and returns it as a dataframe """
     # def readData_csv(self, address):
     #      self.df=pd.read_csv(address,sep=";")
     #      self.Req= self.df['requirement']
     #      return self.Req
     # """ Reads the "requirement" column of an excel file and returns it as a dataframe """
     # def readData_excel(self, address, worksheet):
     #      self.df=pd.read_excel(address,worksheet,encoding="utf-8")
     #      self.Req= self.df['requirement']
     #      return self.Req

     