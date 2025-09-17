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
            const res = await api.get('/admin/branding');
            setBrandingConfig(res.data);

            // Only apply brand title color; do NOT touch Tailwind primary palette
            if (res.data?.primary_color) {
                const base = tinycolor(res.data.primary_color);
                document.documentElement.style.setProperty('--brand-title-color', base.toHexString());
            }

            // Update favicon
            if (res.data?.favicon_url) {
                const favicon = document.querySelector("link[rel*='icon']") as HTMLLinkElement;
                if (favicon) {
                    favicon.href = res.data.favicon_url;
                }
            }
        } catch (e) {
            console.error("Failed to load branding config", e);
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
