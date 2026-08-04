\LoadClass[a4paper,12pt]{article} % Classe du document

%--------------------- Packages ------------------------

\RequirePackage{lmodern} % Polices vectorielles (requis par microtype pour l'expansion de police)
\RequirePackage[french]{babel} % Langue du document
\RequirePackage[utf8]{inputenc} % Caractères spéciaux
\RequirePackage[section]{placeins} % Pour placement de section
\RequirePackage[T1]{fontenc} % Quelques lettres qui sont pas inclus dans UTF-8
\RequirePackage{mathtools} %P aquet pour des équations et symboles mathématiques
\RequirePackage{siunitx} % Pour écrire avec la notation scientifique (Ex.: \num{2e+9})
\RequirePackage{float} % Pour placement d'images
\RequirePackage{graphicx} % Paquet pour insérer des images
\RequirePackage[justification=centering]{caption} % Pour les légendes centralisées
\RequirePackage{subcaption}
\RequirePackage{wallpaper}
\RequirePackage{nomencl}
\RequirePackage{fancyhdr}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref} % Paquet pour insérer légendes dans des sous-figures comme Figure 1a, 1b
\RequirePackage[left=2.5cm,right=2.5cm,top=4cm,bottom=4cm]{geometry} % Configuration de la page
\RequirePackage{tikz}
\usepackage{lipsum}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage[dvipsnames]{xcolor}
\usepackage{etoolbox}
\usepackage{listings} % Package pour afficher du code
\usepackage{ifthen}
\usepackage{booktabs}
\usepackage[acronym]{glossaries}

\makeglossaries

%-------------------- Informations sur le rapport ----------------------

\newcommand{\titre}[1]{\renewcommand{\titre}{#1}}
\newcommand{\eleve}[1]{\renewcommand{\eleve}{#1}}
\newcommand{\mention}[1]{\renewcommand{\mention}{#1}}
\newcommand{\master}[1]{\renewcommand{\master}{#1}}
\newcommand{\filiere}[1]{\renewcommand{\filiere}{#1}}
\newcommand{\dates}[1]{\renewcommand{\dates}{#1}}
\newcommand{\tuteuruniv}[1]{\renewcommand{\tuteuruniv}{#1}}
\newcommand{\tuteurentreprise}[1]{\renewcommand{\tuteurentreprise}{#1}}
\newcommand{\logoentreprisecouverture}[1]{\renewcommand{\logoentreprisecouverture}{#1}}
\newcommand{\logoentreprise}[1]{\renewcommand{\logoentreprise}{#1}}

\newcommand{\rulewidth}{14 cm}

% Abstract
\newcommand{\keywords}[1]{\renewcommand{\keywords}{#1}}
\newcommand{\categories}[1]{\renewcommand{\categories}{#1}}

% Redéfinir l'environnement itemize pour des points colorés dans les listes
\setlist[itemize,1]{label=\textcolor{ENSAIGrey}{\textbullet}}
\setlist[itemize,2]{label=\textcolor{ENSAIBlue}{\textbullet}}
\setlist[itemize,3]{label=\textcolor{SoftENSAIBlue}{\textbullet}}
\setlist[itemize,4]{label=\textcolor{SoftENSAIGrey}{\textbullet}}

% Redéfinir les listes numérotées pour avoir 1.1.1 ...
\renewcommand{\labelenumii}{\arabic{enumi}.\arabic{enumii}}
\renewcommand{\labelenumiii}{\arabic{enumi}.\arabic{enumii}.\arabic{enumiii}}
\renewcommand{\labelenumiv}{\arabic{enumi}.\arabic{enumii}.\arabic{enumiii}.\arabic{enumiv}}

% redéfition de la numérotation des figures
\renewcommand{\thefigure}{\thesection.\arabic{figure}}

% Réinitialisation du compteur de figures à chaque nouvelle section
\makeatletter
\@addtoreset{figure}{section}
\makeatother

% Couleurs personnalisées ---------------------------------------------

\definecolor{ENSAIGrey}{RGB}{99,92,109}
\definecolor{SoftENSAIGrey}{RGB}{238,237,238}
\definecolor{ENSAIBlue}{RGB}{0,141,201}
\definecolor{SoftENSAIBlue}{RGB}{191,230,247}

% Style des blocs de code ---------------------------------------------

\lstdefinestyle{globalstyle}{
    basicstyle=\ttfamily\small,             % Police du code (petite taille, monospace)
    backgroundcolor=\color{SoftENSAIGrey!70!white}, % Couleur du fond
    commentstyle=\color{green!50!black},    % Couleur des commentaires
    keywordstyle=\color{ENSAIBlue!70!black},    % Couleur des mots-clés
    stringstyle=\color{ENSAIBlue},  % Couleur des chaînes de caractères
    numberstyle=\ttfamily\small\color{ENSAIGrey},   % Style des numéros de ligne
    numbers=left,                   % Numéros de ligne à gauche
    frame=single,                   % Cadre autour du code
    breaklines=true,                % Coupe automatiquement les lignes trop longues
    breakatwhitespace=false,        % Ne force pas les coupures de ligne aux espaces
    captionpos=b,                   % Place la légende (caption) en bas du bloc de code
    keepspaces=true,                % Conserve les espaces dans l'affichage du code
    showspaces=false,               % N'affiche pas les espaces sous forme de symboles visibles
    showstringspaces=false,         % N'affiche pas les espaces dans les chaînes de caractères
    showtabs=false,                 % N'affiche pas les tabulations sous forme de symboles visibles
    tabsize=2                       % Définit la taille d'une tabulation à 2 espaces
}

\lstset{style=globalstyle}

% Commande marge ------------------------------------------------------

\newcommand{\fairemarges}{
	\makenomenclature
	\pagestyle{fancy}
	\setlength{\headheight}{21.5pt}
}

% Commande Couverture --------------------------------------------------

\newcommand{\fairecouverture}{
	\begin{titlepage}

		% Couverture du Rapport
		\tikz[remember picture,overlay] \node[opacity=1,inner sep=0pt] at (current page.center){\includegraphics[width=\paperwidth,height=\paperheight]{documents/1e-couverture-ENSAI.png}};

		\vspace{0.6cm}

		\centering % Centraliser le contenu
		\leavevmode

		% Titre du rapport
		{\color{ENSAIGrey} \rule{\rulewidth}{0.2 mm}} \\[0.5 cm]
		{\huge\bfseries \titre \par} \

		{\color{ENSAIGrey} \rule{\rulewidth}{0.2 mm}} \\[0.6 cm]

		% Nom de l'étudiant
		{\scshape\LARGE \eleve \par}
		\vspace{0.6cm}

		% Parcours de l'étudiant (filière/master optionnels : \\ seulement
		% si non vides, sinon 2 \\ consécutifs font planter LaTeX)
		{\scshape \mention
			\ifdefempty{\filiere}{}{\\ \filiere}%
			\ifdefempty{\master}{}{\\ \master}%
		\par}
		\vspace{0.6cm}

		% Dates
		{\large \dates \par}
		\vspace{0.8cm}

		% Représentants de l'étudiant
		\begin{minipage}{0.4\textwidth}
			\begin{flushleft}
				\emph{\textbf{Référent pédagogique :}}\\
				\tuteuruniv \\
			\end{flushleft}
		\end{minipage}
		~
		\begin{minipage}{0.4\textwidth}
			\begin{flushright}
				\emph{\textbf{Tuteurs de projet :}} \\
				\tuteurentreprise \\
			\end{flushright}
		\end{minipage}\\[0.6cm]

		% Logo Entreprise
		\logoentreprisecouverture

		\vfill


	\end{titlepage}
	% Pas de \newpage ici : l'environnement titlepage en insère deja un a sa
	% fermeture (sinon on obtient une page blanche entre la couverture et la
	% suite du document).
}

% Commande Abstract --------------------------------------------------

\newcommand{\abstractsection}[2]{
    \lfoot{\textsl{\textsc{\nouppercase{Résumé / Abstract}}}}
    \renewcommand{\abstractname}{Résumé}
    \begin{abstract}
        #1
    \end{abstract}
    
    \vspace{0.5cm}
    
    \renewcommand{\abstractname}{Abstract}
    \begin{abstract}
        #2
    \end{abstract}
    
    \vspace{1cm}
    \noindent\textbf{Keywords:} \keywords \\
    \noindent\textbf{Categories:} \categories 
    \newpage
    \lfoot{\textsl{\textsc{\nouppercase{\leftmark}}}}
}

% Commande Table des matières -----------------------------------------

\newcommand{\tabledematieres}{
	\tableofcontents
	\newpage
}

% Commande Table des figures -----------------------------------------

\newcommand{\tabledesfigures}{
    \listoffigures
    \newpage
}

% Commande Glossaire -----------------------------------------

\newcommand{\glossaire}[1]{
	\lfoot{\textsl{\textsc{\nouppercase{Glossaire}}}}
	\input{#1} % charge le fichier contenant les entrées

		% Glossaire principal
		\addcontentsline{toc}{section}{\protect\large Glossaire}
		\printglossary[type=main,title={\LARGE \bfseries Glossaire}]

		% Acronymes
		\addcontentsline{toc}{section}{\protect\large Acronymes}
		\printglossary[type=\acronymtype,title={\LARGE \bfseries Liste des acronymes}]
}


% Commande Bibliographie -----------------------------------------

\newcommand{\bibliographie}[2]{
	\lfoot{\textsl{\textsc{\nouppercase{Bibliographie}}}}
	\patchcmd{\thebibliography}{\section*{\refname}}{}{}{}
	\addcontentsline{toc}{section}{\protect\large Bibliographie}
	{\LARGE \bfseries Bibliographie}
	\vspace{0.5cm}
	\bibliographystyle{#2}
	\bibliography{#1}
}

% Commande Annexes -----------------------------------------

\newcommand{\annexes}{
	\lfoot{\textsl{\textsc{\nouppercase{Annexes}}}}
	\appendix
	{\LARGE \bfseries Annexes}
	\addcontentsline{toc}{section}{\protect\large Annexes}
}

% Commande logo haut de page -------------------------------------------

\lhead{\includegraphics[height=0.5cm]{logos/ensai_logo_nom.png}} % Affichage du logo ENSAI au top left de la page

\chead{}

\rhead{\logoentreprise} % Affichage du logo entreprise au top center de la page

% Commande bas de page -------------------------------------------------

\setlength{\footskip}{45pt}

\rfoot{
	\begin{tikzpicture}[remember picture,overlay]
		\node[inner sep=0pt] at ([shift={(7.8cm,-12.3cm)}]current page.center) {\includegraphics[height=2.0cm]{logos/ensai_logo_piedpage.png}};
		\node[text=black, align=center] at ([shift={(8cm,-12.3cm)}]current page.center) {\thepage};
		\draw[line width=0.4pt] (-15.5,0.8) -- (-12,0.8);
	\end{tikzpicture}
}

\cfoot{}

\lfoot{\textsl{\textsc{\nouppercase{\leftmark}}}}