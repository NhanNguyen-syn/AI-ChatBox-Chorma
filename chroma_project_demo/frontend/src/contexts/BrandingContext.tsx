import React, { createContext, useState, useEffect, useContext, ReactNode } from 'react';
import { api } from '../services/api';
import tinycolor from 'tinycolor2';

interface BrandingConfig {
    brand_name: string;
    primary_color: string;
    brand_logo_url: string;
    brand_logo_height: string;
    favicon_url: string;
}

interface BrandingContextType {
    brandingConfig: BrandingConfig | null;
    loadBrandingConfig: () => void;
}

const BrandingContext = createContext<BrandingContextType | undefined>(undefined);

export const BrandingProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [brandingConfig, setBrandingConfig] = useState<BrandingConfig | null>(null);

    const loadBrandingConfig = async () => {
        try {
            // Use public branding endpoint so login page can load without token
            const res = await api.get('/admin/branding/public');
            setBrandingConfig(res.data);

            // Only apply brand title color; do NOT touch Tailwind primary palette
            if (res.data?.primary_color) {
                const base = tinycolor(res.data.primary_color);
                document.documentElement.style.setProperty('--brand-title-color', base.toHexString());

                // Accent stripe colors: fixed green + orange tones per branding preference
                document.documentElement.style.setProperty('--brand-accent-a', '#308748');
                document.documentElement.style.setProperty('--brand-accent-b', '#f6a645');
            }

            // Update favicon
            if (res.data?.favicon_url) {
                const favicon = document.querySelector("link[rel*='icon']") as HTMLLinkElement;
                if (favicon) {
                    favicon.href = res.data.favicon_url;
                }
            }
        } catch (e) {
            // Silently ignore on dev if backend is not running yet
            console.warn("Branding config not available (backend down or no config yet). Using defaults.");
        }
    };

    useEffect(() => {
        loadBrandingConfig();
    }, []);

    return (
        <BrandingContext.Provider value={{ brandingConfig, loadBrandingConfig }}>
            {children}
        </BrandingContext.Provider>
    );
};

export const useBranding = () => {
    const context = useContext(BrandingContext);
    if (context === undefined) {
        throw new Error('useBranding must be used within a BrandingProvider');
    }
    return context;
};
