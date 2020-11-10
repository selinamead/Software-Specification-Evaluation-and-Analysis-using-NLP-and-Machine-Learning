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


     """ selects the number of sentences present in one requirement """
     def select_sentences(self, row, nb="n"):
          # if the module nltk detects less sentences than spacy module
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
          tot =0
          for w in range(len(attribute)):
          # if it is tagged with "$(" (<=> punctuation tag)
               if attribute[w][1]=="$(":
                    tot+=1

          return tot

     def count_comma(self,attribute):
          tot =0
          for w in range(len(attribute)):
               if attribute[w][1]=="$,":
                    tot+=1
          return tot

     def count_weird_words(self, attribute):
          tot =0
          for w in range(len(attribute)):
               if attribute[w][2]=="X":
                    tot+=1

          return tot

     """ Search for specific words in the requirements. When word is beispiel or circa, take abbreviation into account """
     def search_words(self, attribute, search_word):
          if search_word=="beispiel":
               word = ["z.b.","zb","zum beispiel","zb."]
          elif search_word=="circa":
               word = ["circa","ca","ca."]
          else: word = [search_word.lower()]
          tot = 0
          for w in range(len(attribute)):
               if attribute[w][0].lower() in word:
                    tot+=1
          return tot

     """ function that checks the presence of "und" or "oder" in the requirement (double counting with CONJ?)"""
     def count_Copulative_Disjunctive_terms(self, requirement):
          disj_list = ["und","oder"]
          tot = 0
          for w in requirement.split():
               if w.lower() in disj_list:
                    tot+=1
          return tot

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
          tot = 0
          for w in range(len(attribute)):
               if attribute[w][0].lower() in conj_list:
                    tot+=1
          return tot


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
          tot = 0
          for w in range(len(attribute)):
               if attribute[w][2]=="NUM":
                    tot+=1
          return tot

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
          tot = 0
          for w in range(len(attribute)):
               if attribute[w][0].lower() in self.weakwords:
                    tot+=1
               else:
                    w_stem = self.stemmer.stem(attribute[w][0].lower())
                    if w_stem in self.weakwords:
                         tot+=1
          return tot

     def count_weakwords_from_sentence(self, sentence):
          tot = 0
          weak = []
          words = sentence.split()
          for w in words:
               if w.lower() in self.weakwords:
                    tot+=1
                    weak.append(w)
          return tot, weak

     def count_weakwords_from_lemma(self, attribute, stemmer):
          tot = 0
          weak = []
          for w in range(len(attribute)):
               if attribute[w][0].lower() in self.weakwords:
                    tot+=1
                    weak.append(attribute[w][0].lower())
               else:
                    w_stem = stemmer.stem(attribute[w][0].lower())
                    if w_stem in self.weakwords:
                         tot+=1
                         weak.append(w_stem)
          return tot, weak

     """ function that tries to find out if a sentence is written in a passive form 
        as rule : if we have at least one past participe and one auxiliary that is "werden", "wird", "worden" "wurde", then we have a passive sentence
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


     """ counts how many subordinate conjunctions 
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
     
     """ Reads the "requirement" column of an excel file and returns it as a dataframe """
     def readData_csv(self, address):
          self.df=pd.read_csv(address,sep=";")
          self.Req= self.df['requirement']
          return self.Req


     """ Reads the "requirement" column of an excel file and returns it as a dataframe """
     def readData_excel(self, address, worksheet):
          self.df=pd.read_excel(address,worksheet,encoding="utf-8")
          self.Req= self.df['requirement']
          return self.Req

     """ Graphic representation of the requirements vocabulary pool, ordered by words importance (number of appearance) """
     def Corpus_Analysis(self, data, stop):
          count = CountVectorizer()
          tfidf = TfidfTransformer()
          # print option for graphics
          np.set_printoptions(precision=2)
          docs = np.array(data)
          bag = count.fit_transform(docs)
          # create a dictionary with a position for each unique word of the corpus
          Vocabulary = count.vocabulary_
          Bag_array=bag.toarray()
          # vector that count how many time unique word appears in the corpus
          Vocab_count=np.cumsum(Bag_array,axis=0)[-1]
          Vocab_anzahl={}
          # create dictionary: word: nb of occurence in the corpus, remove stop words
          for key, value in Vocabulary.items():
               if key not in stop:
                    # divide by 3, because each req is present 3 times in the initial corpus
                    Vocab_anzahl[key]=Vocab_count[value]/3
                    # order dictionnary by words number of appearances
                    Vocab_sorted=OrderedDict(sorted(Vocab_anzahl.items(), key=lambda t:t[1]))
                    # call the function Vocabolary _Graph to plot ordered vocabulary list
          
          self.Vocabulary_Graph(Vocab_sorted)

     """ function that plots a given dictionary of words with their number of appearances in a text corpus as a bar plot """
     def Vocabulary_Graph(self, Vocab):
          # Find unique words
          x=range(len(np.unique(Vocab.values())))
          height = 125
          text={}
          for i in np.unique(Vocab.values()):
               text[i]=''
          # for each number of appearance, write the list of words having this number of appearance
          for t in range(len(Vocab.keys())):
               # every 10 words, we write a new line
               if t%10==0:
                    text[Vocab.values()[t]]+='<br> '+Vocab.keys()[t]
               else:
                    text[Vocab.values()[t]]+=', '+Vocab.keys()[t]
          trace0=go.Bar(x=np.unique(Vocab.values()),y=height,text=text.values(),orientation="v")
          layout = go.Layout(title="Number of words per Occurence",yaxis=dict(title="Number of Word",showticklabels=True),xaxis=dict(title="Occurence of Words"))
          plot(go.Figure(data=[trace0],layout=layout))

