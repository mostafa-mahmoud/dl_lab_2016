"
" MisoF's .vimrc
" version 2.2 2011-10-24
"

" ======================================================================================================
" For starters, some settings that should be default, but one never knows: {{{
set nocompatible " we want new vim features whenever they are available
" set autoindent
set bs=2         " backspace should work as we expect it to
set history=50   " remember last 50 commands
set ruler        " show cursor position in the bottom line
set number
syntax on        " turn on syntax highlighting if not available by default
" }}}
" ======================================================================================================
" Small tweaks: my preferred indentation, colors, autowrite, status line etc.:  {{{

" currently I prefer indent step 4 and spaces -- tabs are evil and should be avoided
"

set tabstop=4
set softtabstop=4
set shiftwidth=4
set expandtab

function! TABSET()
    set shiftwidth=2
    set tabstop=2
    set softtabstop=2
endfunction

set colorcolumn=80,120

filetype indent on

" by default, if I'm editing text, I want it to wrap
set textwidth=120

set fdm=marker
set commentstring=\ \"\ %s
" nmap <C-V> "+gP
" imap <C-V> <ESC><C-V>i
" vmap <C-C> "+y

nmap <C-A> "+ggVG

vmap ya ggVG"+y
"vmap aa <ESC>+ggVG
map m :make

" showing white spaces, to disable use ":set nolist"
:set listchars=tab:>-,trail:~,extends:>,precedes:<
set nolist

" my terminal is dark, use an appropriate colorscheme
"set background=dark
" use the following to force black background if necessary:
" highlight Normal guibg=black guifg=white ctermbg=black ctermfg=white 

" automatically save before each make/execute command
set autowrite

" if I press <tab> in command line, show me all options if there is more than one
set wildmenu

" y and d put stuff into system clipboard (so that other apps can see it)
"set clipboard+=unnamed
"set clipboard+=unnamedplus

set clipboard=unnamedplus

" adjust timeout for mapped commands: 200 milliseconds should be enough for everyone
set timeout
set timeoutlen=200

" an alias to convert a file to html, using vim syntax highlighting
" command ConvertToHTML so $VIMRUNTIME/syntax/2html.vim

" text search settings
set incsearch  " show the first match already while I type
set ignorecase
set smartcase  " only be case-sensitive if I use uppercase in my query
set hlsearch " I hate when half of the text lights up

" enough with the @@@s, show all you can if the last displayed line is too long
set display+=lastline
" show chars that cannot be displayed as <13> instead of ^M
set display+=uhex

" status line: we want it at all times -- white on blue, with ASCII code of the current letter
set statusline=%<%f%h%m%r%=char=%b=0x%B\ \ %l,%c%V\ %P
set laststatus=2
set highlight+=s:MyStatusLineHighlight
highlight MyStatusLineHighlight ctermbg=darkblue ctermfg=white

" tab line: blue as well to fit the theme
" (this is what appears instead of the status line when you use <tab> in command mode)
highlight TabLine ctermbg=darkblue ctermfg=gray
highlight TabLineSel ctermbg=darkblue ctermfg=yellow
highlight TabLineFill ctermbg=darkblue ctermfg=darkblue

" }}}
" ======================================================================================================
"" Different functions: {{{{
"" Colours in xterm.
map <F3> :se t_Co=16<C-M>:se t_AB=<C-V><ESC>[%?%p1%{8}%<%t%p1%{40}%+%e%p1%{92}%+%;%dm<C-V><C-M>:se t_AF=<C-V><ESC>[%?%p1%{8}%<%t%p1%{30}%+%e%p1%{82}%+%;%dm<C-V><C-M>

"" Toggle between .h and .cpp with F4.
function! ToggleBetweenHeaderAndSourceFile()
  let bufname = bufname("%")
  let ext = fnamemodify(bufname, ":e")
  if ext == "h"
    let ext = "cpp"
  elseif ext == "cpp"
    let ext = "h"
  else
    return
  endif
  let bufname_new = fnamemodify(bufname, ":r") . "." . ext
  let bufname_alt = bufname("#")
  if bufname_new == bufname_alt
    execute ":e#"
  else
    execute ":e " . bufname_new
  endif
endfunction
map <silent> <F4> :call ToggleBetweenHeaderAndSourceFile()<CR>


"" Toggle encoding with F12.
function! ToggleEncoding()
  if &encoding == "latin1"
    set encoding=utf-8
  elseif &encoding == "utf-8"
    set encoding=latin1
  endif
endfunction
map <silent> <F12> :call ToggleEncoding()<CR>

"" Cycling through Windows quicker.
map <C-M> <C-W>j<C-W>_ 
map <C-K> <C-W>k<C-W>_ 
map <A-Down>  <C-W><Down><C-W>_
map <A-Up>    <C-W><Up><C-W>_
map <A-Left>  <C-W><Left><C-W>|
map <A-Right> <C-W><Right><C-W>|

" <Tab> at the end of a word should attempt to complete it using tokens from the current file:
function! My_Tab_Completion()
    if col('.')>1 && strpart( getline('.'), col('.')-2, 3 ) =~ '^\w'
        return "\<C-P>"
    else
        return "\<Tab>"
endfunction
inoremap <Tab> <C-R>=My_Tab_Completion()<CR>
" }}}
" ======================================================================================================
" Specific settings for specific filetypes:  {{{

" usual policy: if there is a Makefile present, :mak calls make, otherwise we define a command to compile the filetype

" LaTeX
function! TEXSET()
  set makeprg=if\ \[\ -f\ \"Makefile\"\ \];then\ make\ $*;else\ if\ \[\ -f\ \"makefile\"\ \];then\ make\ $*;else\ pdflatex\ %;fi;fi
  set errorformat=%f:%l:\ %m
  set spell
  set textwidth=0
  set nowrap
endfunction

function! JAVACUP()
  set makeprg=if\ \[\ -f\ \"Makefile\"\ \];then\ make\ $*;else\ if\ \[\ -f\ \"makefile\"\ \];then\ make\ $*;else\ java\ java_cup.Main\ %;fi;fi
  set errorformat=%f:%l:\ %m
  set spell
endfunction

function! GRAPHVIZSET()
  set makeprg=if\ \[\ -f\ \"Makefile\"\ \];then\ make\ $*;else\ if\ \[\ -f\ \"makefile\"\ \];then\ make\ $*;else\ dot\ -Tpng\ -o\ %.png\ %;fi;fi
  set errorformat=%f:%l:\ %m
endfunction

function! COQSET()
  set makeprg=if\ \[\ -f\ \"Makefile\"\ \];then\ make\ $*;else\ if\ \[\ -f\ \"makefile\"\ \];then\ make\ $*;else\ coq\ %;fi;fi
  set errorformat=%f:%l:\ %m
endfunction

" C/C++:
function! CSET()
  abbr iio #include <stdio.h>
  abbr icsr #include <string.h>
  abbr ilib #include <stdlib.h>
  abbr iass #include <assert.h>
  abbr imath #include <math.h>
  abbr ll long long
  abbr ull unsigned long long
  abbr i64 int64
  abbr DEFSCAN #define next_int() ({int __t; scanf("%d", &__t); __t;})
  abbr DEFIT #define for_each(s, v) for (__typeof((v).begin()) s = (v).begin(); s != (v).end(); ++s)
  abbr DEFMEM #define mem(s, v) memset(s, v, sizeof(s))
  abbr MAIN int main() {}
  syntax match Type "\<int64\>"
  syntax match Type "\<point\>"
  syntax match Type "\<pair\>"
  syntax match Type "\<Node\>"
  syntax match Operator "\<for_each\>"
  syntax match Function "\<[fs]*scanf\>"
  syntax match Function "\<[fs]*printf\>"
  syntax match Function "\<max\>"
  syntax match Function "\<min\>"
  syntax match Function "\<abs\>"
  syntax match Function "\<fabs\>"
  syntax match Function "\<sort\>"
  syntax match Function "\<gcd\>"
  syntax match Function "\<assert\>"
  syntax match Function "\<strlen\>"
  syntax match Function "\<getchar\>"
  syntax match Function "\<puts\>"
  syntax match Function "\<size\>"
  syntax match Function "\<memset\>"
  syntax match Function "\<count\>"
  syntax match Function "\<get\>"
  syntax match Function "\<mem\>"
  syntax match Function "\<freopen\>"
  syntax match Constant "\<INF\>"
  syntax match Constant "\<MAX\>"
  syntax match Constant "\<EPS\>"
  syntax match Constant "\<PI\>"
  syntax match Warning "\<cost\>"
  set makeprg=if\ \[\ -f\ \"Makefile\"\ \];then\ make\ $*;else\ if\ \[\ -f\ \"makefile\"\ \];then\ make\ $*;else\ gcc\ -O2\ -pthread\ -g\ -Wall\ -W\ -lm\ -o%.bin\ %;fi;fi
  set errorformat=%f:%l:\ %m
  set cindent
  set tw=0
  set nowrap
endfunction

" C++
function! CPPSET()
  abbr vi vector<int>
  abbr vvi vector<vector <int> >
  abbr vs vector<string>
  abbr vll vector<long long>
  abbr vvll vector<vector<long long> >
  abbr msi map<string, int>
  abbr mii map<int, int>
  abbr pii pair<int, int>
  abbr vpii vector<pair<int, int> >
  abbr si set<int>
  abbr ll long long
  abbr ull unsigned long long
  abbr sz size()
  abbr PB push_back
  abbr MP make_pair
  abbr iio #include <cstdio>
  abbr icstr #include <cstring>
  abbr ialgo #include <algorithm>
  abbr inum #include <numeric>
  abbr imath #include <cmath>
  abbr iiost #include <iostream>
  abbr isst #include <sstream>
  abbr ivec #include <vector>
  abbr istr #include <string>
  abbr imap #include <map>
  abbr iset #include <set>
  abbr iass #include <cassert>
  abbr ibits #include <bitset>
  abbr iqueue #include <queue>
  abbr icmplx #include <complex>
  abbr i64 int64
  abbr STD using namespace std;
  abbr DEFPII typedef complex <int> point;
  abbr DEFPDD typedef complex <double> point;
  abbr DEFSCAN #define next_int() ({int __t; scanf("%d", &__t); __t;})
  abbr DEFIT #define for_each(s, v) for (__typeof((v).begin()) s = (v).begin(); s != (v).end(); ++s)
  abbr popcnt __builtin_popcount
  abbr DEFALL #define all(c) (c).begin(), (c).end()
  abbr DEFRALL #define rall(c) (c).rbegin(), (c).rend()
  abbr DEFMEM #define mem(s, v) memset(s, v, sizeof(s))
  abbr MAIN int main() {}
  syntax match Type "\<int64\>"
  syntax match Type "\<string\>"
  syntax match Type "\<[io]*stringstream\>"
  syntax match Type "\<vector\>"
  syntax match Type "\<map\>"
  syntax match Type "\<set\>"
  syntax match Type "\<multiset\>"
  syntax match Type "\<point\>"
  syntax match Type "\<pair\>"
  syntax match Type "\<priority_queue\>"
  syntax match Type "\<queue\>"
  syntax match Type "\<deque\>"
  syntax match Type "\<Node\>"
  syntax match Operator "\<for_each\>"
  syntax match Function "\<[fs]*scanf\>"
  syntax match Function "\<count\>"
  syntax match Function "\<get\>"
  syntax match Function "\<[fs]*printf\>"
  syntax match Function "\<max\>"
  syntax match Function "\<min\>"
  syntax match Function "\<abs\>"
  syntax match Function "\<fabs\>"
  syntax match Function "\<sort\>"
  syntax match Function "\<gcd\>"
  syntax match Function "\<__gcd\>"
  syntax match Function "\<assert\>"
  syntax match Function "\<strlen\>"
  syntax match Function "\<getchar\>"
  syntax match Function "\<puts\>"
  syntax match Function "\<size\>"
  syntax match Function "\<memset\>"
  syntax match Function "\<mem\>"
  syntax match Function "\<freopen\>"
  syntax match Constant "\<INF\>"
  syntax match Constant "\<MAX\>"
  syntax match Constant "\<EPS\>"
  syntax match Constant "\<PI\>"
  syntax match Operator "\<endl\>"
  syntax match Operator "\<first\>"
  syntax match Operator "\<second\>"
  syntax match Warning "\<cost\>"
  set list
  set shiftwidth=2
  set tabstop=2
  set expandtab
  set softtabstop=2
  set textwidth=80
  set makeprg=if\ \[\ -f\ \"Makefile\"\ \];then\ make\ $*;else\ if\ \[\ -f\ \"makefile\"\ \];then\ make\ $*;else\ g++\ %\ -O2\ -g\ -std=c++11\ -Wall\ -W\ -o%.bin;fi;fi
  set cindent
  set tw=0
  set nowrap
endfunction

" Java
function! JAVASET()
  abbr throwIO throws IOException
  abbr bfrdr BufferedReader std = new BufferedReader(new InputStreamReader(System.in));
  abbr ibigint import java.math.BigInteger;
  abbr ioe import java.io.IOException;
  abbr ibfrdr import java.io.BufferedReader;
  abbr iinpst import java.io.InputStreamReader;
  abbr sysout System.out.println
  abbr sysoutf System.out.printf
  abbr MAIN public static void main(String[] args)
  abbr ln length
  syntax match Type "\<String\>"
  set textwidth=80
  set makeprg=if\ \[\ -f\ \"Makefile\"\ \];then\ make\ $*;else\ if\ \[\ -f\ \"makefile\"\ \];then\ make\ $*;else\ javac\ -g\ %;fi;fi
  set errorformat=%f:%l:\ %m
  set cindent
  set tw=0
  set nowrap
endfunction

" Pascal
function! PPSET()
  set makeprg=if\ \[\ -f\ \"Makefile\"\ \];then\ make\ $*;else\ if\ \[\ -f\ \"makefile\"\ \];then\ make\ $*;else\ fpc\ -g\ -O2\ -o\%.bin\ %;fi;fi
  set errorformat=%f:%l:\ %m
  set tw=0
  set nowrap
endfunction

" C# Settings
function! CSSET()
  set makeprg=if\ \[\ -f\ \"Makefile\"\ \];then\ make\ $*;else\ if\ \[\ -f\ \"makefile\"\ \];then\ make\ $*;else\ mono-csc\ -out\:\%.bin\ %;fi;fi
  set errorformat=%f:%l:\ %m
  set tw=0
  set nowrap
endfunction

" vim scripts
function! VIMSET()
  set tw=0
  set nowrap
  set comments+=b:\"
endfunction

" Makefile
function! MAKEFILESET()
  set tw=0
  set nowrap
  " in a Makefile we need to use <Tab> to actually produce tabs
  set noet
  set sts=8
  iunmap <Tab>
endfunction

" HTML/PHP
function! HTMLSET()
  set tw=0
  set nowrap
endfunction

" Asymptote
function! ASYSET()
  runtime asy.vim " find this somewhere and place it into ~/.vim/ for syntax hl to work
  set tw=0
  set nowrap
  set makeprg=asy\ -noV\ -fpdf\ %\ -o\ %.pdf
  set errorformat=%f:\ %l.%c:\ %m
endfunction

" Python
function! PYSET()
  set list
  set shiftwidth=4
  set tabstop=4
  set expandtab
  set softtabstop=4
  set textwidth=80
  set nowrap
  syntax match Constant "self"
endfunction

" Ruby
function! RBSET()
  set list
  set shiftwidth=2
  set tabstop=2
  set expandtab
  set softtabstop=2
  set textwidth=0
  set nowrap
endfunction

" Asymptote does not get recognized by default, fix it
augroup filetypedetect
autocmd BufNewFile,BufRead *.asy setfiletype asy
augroup END
filetype plugin on

" Autocommands for all languages:
autocmd FileType vim    call VIMSET()
autocmd FileType cup    call JAVACUP()
autocmd FileType c      call CSET()
autocmd FileType C      call CPPSET()
autocmd FileType cc     call CPPSET()
autocmd FileType cpp    call CPPSET()
autocmd FileType java   call JAVASET()
autocmd FileType tex    call TEXSET()
autocmd FileType pascal call PPSET()
autocmd FileType make   call MAKEFILESET()
autocmd FileType html   call HTMLSET()
autocmd FileType php    call HTMLSET()
autocmd FileType asy    call ASYSET()
autocmd FileType python call PYSET()
autocmd FileType cs		call CSSET()
autocmd FileType ruby	call RBSET()
autocmd FileType dot    call GRAPHVIZSET()
autocmd FileType v      call COQSET()
"au BufRead,BufNewFile *.v   set filetype=coq

"" But TABs are needed in Makefiles
au BufNewFile,BufReadPost Makefile se noexpandtab

" }}}
" ======================================================================================================
" finally, tell the folds to fold on file open
" autocmd GUIEnter * winpos 0 0 | set lines=9999 columns=9999
" colorscheme default
set guifont=UbuntuMono\ 13

" set runtimepath^=~/.vim/bundle/ctrlp.vim
