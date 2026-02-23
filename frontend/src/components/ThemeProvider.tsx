import React, { createContext, useContext, useEffect, useState } from 'react';

type Theme = 'dark' | 'light';

interface ThemeContextType {
      theme: Theme;
      toggleTheme: (event?: React.MouseEvent) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
      const [theme, setTheme] = useState<Theme>('dark');

      // Mount logic: enforce dark initially
      useEffect(() => {
            document.documentElement.classList.add('dark');
            document.documentElement.classList.remove('light');
      }, []);

      const toggleTheme = async (event?: React.MouseEvent) => {
            const isDark = theme === 'dark';
            const nextTheme = isDark ? 'light' : 'dark';

            // Wait to finish if another transition is running
            if (!document.startViewTransition) {
                  // Fallback for browsers that don't support View Transitions API
                  updateDOM(nextTheme);
                  return;
            }

            // Get coordinates for the epicenter of the circle
            const x = event?.clientX ?? window.innerWidth / 2;
            const y = event?.clientY ?? window.innerHeight / 2;
            const endRadius = Math.hypot(
                  Math.max(x, window.innerWidth - x),
                  Math.max(y, window.innerHeight - y)
            );

            const transition = document.startViewTransition(() => {
                  updateDOM(nextTheme);
            });

            transition.ready.then(() => {
                  const clipPath = [
                        `circle(0px at ${x}px ${y}px)`,
                        `circle(${endRadius}px at ${x}px ${y}px)`,
                  ];

                  // Animate native DOM via Web Animations API 
                  document.documentElement.animate(
                        {
                              clipPath: isDark ? clipPath : [...clipPath].reverse(),
                        },
                        {
                              duration: 600,
                              easing: 'ease-in-out',
                              // Indicate which pseudo-element gets animated
                              pseudoElement: isDark
                                    ? '::view-transition-new(root)'
                                    : '::view-transition-old(root)',
                        }
                  );
            });
      };

      const updateDOM = (nextTheme: Theme) => {
            document.documentElement.classList.remove(theme);
            document.documentElement.classList.add(nextTheme);
            setTheme(nextTheme);
      };

      return (
            <ThemeContext.Provider value={{ theme, toggleTheme }}>
                  {children}
            </ThemeContext.Provider>
      );
}

export function useTheme() {
      const context = useContext(ThemeContext);
      if (!context) {
            throw new Error('useTheme must be used within a ThemeProvider');
      }
      return context;
}
