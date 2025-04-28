import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from './providers';
import ThemeSwitch from './ThemeSwitch';

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ChatBot UI",
  description: "A sleek and responsive chat interface",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased h-full bg-[var(--background)] text-[var(--foreground)] transition-colors duration-300`}>
        <Providers>
          <main className="min-h-screen flex items-center justify-center">
            <div className="w-full max-w-3xl bg-[var(--card-background)] border border-[var(--border)] rounded-xl shadow-xl overflow-hidden relative transition-colors duration-300">
              
              {/* Theme Switcher */}
              <div className="absolute top-4 right-4 z-10">
                <ThemeSwitch />
              </div>
  
              {children}
            </div>
          </main>
        </Providers>
      </body>
    </html>
  );
}

