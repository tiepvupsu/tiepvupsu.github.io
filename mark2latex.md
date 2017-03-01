`filename = 2016-12-28-linearregression.markdown`
1. ## -> section 
```markdown
<a name="-gioi-thieu"></a>

## 1. Giới thiệu
```

will be: 

```latex 
\section{Giới thiệu}
\label{sec:linearregression#-gioi-thieu}
```

2. ### -> subsection 
```markdown
<a name="dang-cua-linear-regression"></a>

### Dạng của Linear Regression 
```

will be: 

```latex
\subsection{Dạng của Linear Regression}
\label{sub:linearregression_dang-cua-linear-regression}
```

3. `\\(` -> `$` 
4. `\\)` -> `$` 

5. Nếu sau `\\[` là `\begin{eqnarray}` thì bỏ `\\[` và `\\]` tương ứng sau `\end{eqnarray}`

6. Links 
[Display name](http...) -> \href{http...}{Display name}

7. Table (difficult)

8. Figures 


9. Codes 

