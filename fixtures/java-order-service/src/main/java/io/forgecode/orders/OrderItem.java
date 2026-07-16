package io.forgecode.orders;

import java.math.BigDecimal;

public final class OrderItem {
    private final int quantity;
    private final BigDecimal unitPrice;

    public OrderItem(int quantity, BigDecimal unitPrice) {
        if (quantity < 0) {
            throw new IllegalArgumentException();
        }
        if (unitPrice == null || unitPrice.signum() < 0) {
            throw new IllegalArgumentException();
        }
        this.quantity = quantity;
        this.unitPrice = unitPrice;
    }

    public int getQuantity() {
        return quantity;
    }

    public BigDecimal getUnitPrice() {
        return unitPrice;
    }
}
